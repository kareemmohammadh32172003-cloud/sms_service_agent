"""
=====================================================================
Telegram bot - the interface each user actually interacts with.

Commands:
    /start           - registers you and gives you your personal webhook URL
    /today           - today's transactions
    /month           - this month's summary
    /lastmonth       - last month's summary
    /budget <cat> <amount>   - set a monthly budget for a category
    /budgetstatus    - see how you're doing against your budgets

You can also just paste a raw SMS directly into the chat and the bot
will record it, same as the automated webhook would.

.env needs:
    TELEGRAM_BOT_TOKEN=123456:ABC...   (get this from @BotFather)
    PUBLIC_WEBHOOK_BASE=https://your-app.up.railway.app

Install:
    pip install python-telegram-bot

Run:
    python telegram_bot.py
"""

import os
import io
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

import finance_core as core

load_dotenv()

logging.basicConfig(level=logging.INFO)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PUBLIC_BASE = os.getenv("PUBLIC_WEBHOOK_BASE", "https://your-app.up.railway.app")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    name = update.effective_user.first_name or ""
    user = core.get_or_create_user(chat_id, display_name=name)

    webhook_url = f"{PUBLIC_BASE}/sms-webhook/{user['api_token']}"
    await update.message.reply_text(
        f"أهلاً {name}!\n\n"
        f"ده الرابط الخاص بيك، حطه في تطبيق SMS Forwarder بتاعك:\n"
        f"{webhook_url}\n\n"
        f"أو ممكن كمان تلصق أي رسالة SMS هنا مباشرة وأنا هسجلها.\n\n"
        f"وكمان تقدر تقولي بصوتك أو بكلامك العادي على أي مصروف كاش (زي "
        f"'دفعت 50 جنيه تاكسي')، مش لازم يكون رسالة بنك رسمية.\n\n"
        f"الأوامر المتاحة: /today  /month  /lastmonth  /budgetstatus  /fix  /chart  /undo  /subscriptions  /projection"
    )


async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = core.get_or_create_user(update.effective_chat.id)
    result = core.query_transactions(user["id"], period="today")
    await update.message.reply_text(result)


async def month(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = core.get_or_create_user(update.effective_chat.id)
    result = core.query_transactions(user["id"], period="this_month")
    await update.message.reply_text(result)


async def lastmonth(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = core.get_or_create_user(update.effective_chat.id)
    result = core.query_transactions(user["id"], period="last_month")
    await update.message.reply_text(result)


async def budget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = core.get_or_create_user(update.effective_chat.id)
    if len(context.args) < 2:
        await update.message.reply_text("استخدم الصيغة: /budget food 2000")
        return
    category, amount = context.args[0], context.args[1]
    try:
        amount = float(amount)
    except ValueError:
        await update.message.reply_text("المبلغ لازم يكون رقم.")
        return
    result = core.set_budget(user["id"], category, amount)
    await update.message.reply_text(result)


async def budget_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = core.get_or_create_user(update.effective_chat.id)
    result = core.check_budget_status(user["id"])
    await update.message.reply_text(result)


async def fix(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = core.get_or_create_user(update.effective_chat.id)
    if not context.args:
        await update.message.reply_text(
            "استخدم الصيغة: /fix food\n"
            f"الفئات المتاحة: {', '.join(core.VALID_CATEGORIES)}"
        )
        return
    new_category = context.args[0].lower()
    result = core.correct_last_transaction_category(user["id"], new_category)
    await update.message.reply_text(result)


async def undo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = core.get_or_create_user(update.effective_chat.id)
    result = core.delete_last_transaction(user["id"])
    await update.message.reply_text(result)


async def subscriptions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = core.get_or_create_user(update.effective_chat.id)
    result = core.detect_recurring_subscriptions(user["id"])
    await update.message.reply_text(result)


async def projection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = core.get_or_create_user(update.effective_chat.id)
    result = core.project_month_end_spending(user["id"])
    await update.message.reply_text(result)


PERIOD_ALIASES = {
    "today": "today", "month": "this_month", "lastmonth": "last_month",
}


async def chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = core.get_or_create_user(update.effective_chat.id)
    period_arg = context.args[0].lower() if context.args else "month"
    period = PERIOD_ALIASES.get(period_arg, "this_month")

    image_bytes = core.build_expense_pie_chart(user["id"], period=period)
    if image_bytes is None:
        await update.message.reply_text("مفيش مصاريف مسجلة للفترة دي عشان أرسم بيها.")
        return

    await update.message.reply_photo(photo=io.BytesIO(image_bytes))


# Per-user conversation memory for the interactive agent, so it can
# handle follow-up questions ("قولي العمليات دي") rather than treating
# every typed message as a fresh, isolated request. In-memory only -
# resets on redeploy, which is fine (worst case: agent forgets recent
# chat context, but all recorded transactions themselves are safe in
# Supabase regardless).
_conversation_histories = {}


async def handle_raw_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Any plain text that isn't a command goes through the interactive
    agent - it can record a pasted SMS (same as before) AND answer
    follow-up questions like 'list these transactions' or 'how much
    did I spend on food', because unlike the one-way webhook path it's
    allowed to use query_transactions and hold a conversation."""
    user = core.get_or_create_user(update.effective_chat.id)
    chat_id = update.effective_chat.id
    history = _conversation_histories.setdefault(chat_id, [])

    result = core.run_finance_agent(update.message.text, history, user_id=user["id"])
    await update.message.reply_text(result)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Voice notes get transcribed then run through the exact same
    conversational pipeline as typed messages - the easiest way to
    log cash spending (taxi, street food, small purchases) that no
    bank SMS will ever capture."""
    user = core.get_or_create_user(update.effective_chat.id)
    chat_id = update.effective_chat.id

    voice_file = await context.bot.get_file(update.message.voice.file_id)
    audio_bytes = bytes(await voice_file.download_as_bytearray())

    try:
        transcribed_text = core.transcribe_voice(audio_bytes)
    except Exception as e:
        await update.message.reply_text(f"معرفتش أفهم الرسالة الصوتية، جرب تاني أو اكتبها. ({e})")
        return

    if not transcribed_text.strip():
        await update.message.reply_text("معرفتش أسمع حاجة واضحة في الرسالة، جرب تاني.")
        return

    history = _conversation_histories.setdefault(chat_id, [])
    result = core.run_finance_agent(transcribed_text, history, user_id=user["id"])
    await update.message.reply_text(f"🎤 سمعت: \"{transcribed_text}\"\n\n{result}")


def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("today", today))
    app.add_handler(CommandHandler("month", month))
    app.add_handler(CommandHandler("lastmonth", lastmonth))
    app.add_handler(CommandHandler("budget", budget))
    app.add_handler(CommandHandler("budgetstatus", budget_status))
    app.add_handler(CommandHandler("fix", fix))
    app.add_handler(CommandHandler("undo", undo))
    app.add_handler(CommandHandler("subscriptions", subscriptions))
    app.add_handler(CommandHandler("projection", projection))
    app.add_handler(CommandHandler("chart", chart))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_raw_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    app.run_polling()


if __name__ == "__main__":
    main()