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
        f"الأوامر المتاحة: /today  /month  /lastmonth  /budgetstatus"
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


async def handle_raw_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Any plain text that isn't a command is treated like a forwarded SMS."""
    user = core.get_or_create_user(update.effective_chat.id)
    result = core.process_incoming_sms(update.message.text, user_id=user["id"])
    await update.message.reply_text(result)


def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("today", today))
    app.add_handler(CommandHandler("month", month))
    app.add_handler(CommandHandler("lastmonth", lastmonth))
    app.add_handler(CommandHandler("budget", budget))
    app.add_handler(CommandHandler("budgetstatus", budget_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_raw_message))

    app.run_polling()


if __name__ == "__main__":
    main()