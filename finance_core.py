"""
=====================================================================
Personal Finance Agent - Core Engine (multi-tenant Supabase version)
=====================================================================

Same Agent Loop pattern as before, but every read/write is now
scoped to a user_id, so many people can safely share the same
deployment without seeing each other's data.

.env needs:
    GROQ_API_KEY=gsk_...
    SUPABASE_URL=https://xxxxx.supabase.co
    SUPABASE_KEY=eyJhbGc...
    TELEGRAM_BOT_TOKEN=123456:ABC...   (added for the bot / monthly job)
"""

import os
import io
import json
import secrets
import time
import calendar
import requests
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
from groq import Groq
from supabase import create_client
import matplotlib
matplotlib.use("Agg")  # headless server, no display available
import matplotlib.pyplot as plt

load_dotenv()

groq_client = Groq()
MODEL_NAME = "openai/gpt-oss-120b"

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

VALID_CATEGORIES = [
    "food", "transport", "bills", "shopping", "entertainment",
    "health", "transfer", "salary", "other"
]


# =================================================================
# User management - links a Telegram account to a private webhook
# =================================================================

def get_or_create_user(telegram_chat_id: int, display_name: str = "") -> dict:
    """Returns the user row for this Telegram chat, creating one
    (with a fresh api_token) the first time they say /start."""
    existing = supabase.table("users").select("*") \
        .eq("telegram_chat_id", telegram_chat_id).execute().data
    if existing:
        return existing[0]

    token = secrets.token_urlsafe(24)
    row = supabase.table("users").insert({
        "telegram_chat_id": telegram_chat_id,
        "api_token": token,
        "display_name": display_name,
    }).execute().data
    return row[0]


def get_user_by_token(token: str) -> dict | None:
    """Looks up which user a webhook request belongs to, based on
    the token in their personal webhook URL."""
    rows = supabase.table("users").select("*").eq("api_token", token).execute().data
    return rows[0] if rows else None


def list_all_users() -> list[dict]:
    """Used by the monthly summary job to loop over everyone."""
    return supabase.table("users").select("*").execute().data


# =================================================================
# Tools - same signatures as before, all scoped by user_id
# =================================================================

DUPLICATE_WINDOW_MINUTES = 60


def _is_likely_duplicate(user_id: str, amount: float, party: str, type: str) -> bool:
    """Guards against the same SMS being forwarded twice, without the
    false-positive risk of blocking two genuinely different people who
    happen to send the same amount. Same amount + same direction within
    the window, AND the party names are similar (one contains the
    other, e.g. 'X' vs 'X جماعة' from the AI extracting a retry
    slightly differently) - not just any two unrelated names."""
    cutoff = (datetime.utcnow() - timedelta(minutes=DUPLICATE_WINDOW_MINUTES)).isoformat()
    rows = supabase.table("transactions").select("id, party") \
        .eq("user_id", user_id) \
        .eq("amount", amount) \
        .eq("type", type) \
        .gte("created_at", cutoff) \
        .execute().data

    party_norm = (party or "").strip().lower()
    for r in rows:
        existing_norm = (r.get("party") or "").strip().lower()
        if not party_norm and not existing_norm:
            return True  # both missing a party (e.g. ATM withdrawals) - treat as duplicate
        if party_norm and existing_norm and (party_norm in existing_norm or existing_norm in party_norm):
            return True
    return False


def add_transaction(user_id: str, amount: float, category: str, type: str,
                     party: str = "", raw_text: str = "", txn_date: str = None) -> str:
    if category not in VALID_CATEGORIES:
        category = "other"
    if type not in ("expense", "income"):
        return f"Error: type must be 'expense' or 'income', got '{type}'"

    if _is_likely_duplicate(user_id, amount, party, type):
        return "Skipped: this looks like a duplicate of a transaction recorded recently."

    txn_date = txn_date or date.today().isoformat()

    supabase.table("transactions").insert({
        "user_id": user_id,
        "txn_date": txn_date,
        "amount": amount,
        "party": party,
        "category": category,
        "type": type,
        "raw_text": raw_text,
    }).execute()

    sign = "-" if type == "expense" else "+"
    return f"Recorded: {sign}{amount} EGP | {category} | {party or 'N/A'}"


def get_last_transaction(user_id: str) -> dict | None:
    rows = supabase.table("transactions").select("*") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .limit(1).execute().data
    return rows[0] if rows else None


def correct_last_transaction_category(user_id: str, new_category: str) -> str:
    if new_category not in VALID_CATEGORIES:
        return f"Unknown category '{new_category}'. Valid options: {', '.join(VALID_CATEGORIES)}"

    last = get_last_transaction(user_id)
    if not last:
        return "You don't have any recorded transactions yet."

    old_category = last["category"]
    supabase.table("transactions").update({"category": new_category}).eq("id", last["id"]).execute()

    sign = "-" if last["type"] == "expense" else "+"
    return (f"Fixed: {sign}{last['amount']} EGP | {last['party'] or 'N/A'} "
            f"moved from '{old_category}' to '{new_category}'")


def delete_last_transaction(user_id: str) -> str:
    last = get_last_transaction(user_id)
    if not last:
        return "You don't have any recorded transactions yet."

    supabase.table("transactions").delete().eq("id", last["id"]).execute()

    sign = "-" if last["type"] == "expense" else "+"
    return f"Deleted: {sign}{last['amount']} EGP | {last['category']} | {last['party'] or 'N/A'}"


def query_transactions(user_id: str, period: str = "this_month", category: str = None) -> str:
    now = datetime.now()

    query = supabase.table("transactions").select("*").eq("user_id", user_id)

    if period == "this_month":
        query = query.gte("txn_date", f"{now.strftime('%Y-%m')}-01")
    elif period == "today":
        query = query.eq("txn_date", date.today().isoformat())
    elif period == "last_month":
        last_month = now.month - 1 or 12
        year = now.year if now.month > 1 else now.year - 1
        query = query.gte("txn_date", f"{year}-{last_month:02d}-01") \
                      .lt("txn_date", f"{now.strftime('%Y-%m')}-01")

    if category:
        query = query.eq("category", category)

    rows = query.execute().data

    if not rows:
        return f"No transactions found for period '{period}'" + (f" in category '{category}'" if category else "")

    # Group individual transactions under their category, so the
    # user sees each item (party, amount) not just a category total.
    by_category = {}
    for r in rows:
        key = (r["category"], r["type"])
        by_category.setdefault(key, []).append(r)

    lines = []
    total_expense = 0.0
    total_income = 0.0
    for (cat, type_), txns in sorted(by_category.items(), key=lambda kv: -sum(t["amount"] for t in kv[1])):
        cat_total = sum(t["amount"] for t in txns)
        lines.append(f"\n{cat} ({type_}) - {cat_total:.2f} EGP total:")
        for t in sorted(txns, key=lambda t: t.get("created_at", ""), reverse=True):
            party = t.get("party") or "N/A"
            lines.append(f"  • {t['amount']:.2f} EGP - {party}")
        if type_ == "expense":
            total_expense += cat_total
        else:
            total_income += cat_total

    summary = f"Period: {period}" + "\n".join(lines)
    summary += f"\n\nTotal expenses: {total_expense:.2f} EGP | Total income: {total_income:.2f} EGP"
    return summary


def get_expense_category_totals(user_id: str, period: str = "this_month") -> dict:
    """Same filtering logic as query_transactions, but returns raw
    {category: total_amount} for expenses only - used to draw charts."""
    now = datetime.now()
    query = supabase.table("transactions").select("*").eq("user_id", user_id).eq("type", "expense")

    if period == "this_month":
        query = query.gte("txn_date", f"{now.strftime('%Y-%m')}-01")
    elif period == "today":
        query = query.eq("txn_date", date.today().isoformat())
    elif period == "last_month":
        last_month = now.month - 1 or 12
        year = now.year if now.month > 1 else now.year - 1
        query = query.gte("txn_date", f"{year}-{last_month:02d}-01") \
                      .lt("txn_date", f"{now.strftime('%Y-%m')}-01")

    rows = query.execute().data
    totals = {}
    for r in rows:
        totals[r["category"]] = totals.get(r["category"], 0.0) + r["amount"]
    return totals


def build_expense_pie_chart(user_id: str, period: str = "this_month") -> bytes | None:
    """Renders a PNG pie chart of expenses by category. Returns None
    if there's nothing to chart (so the caller can send a plain
    'no data' message instead of a blank image)."""
    totals = get_expense_category_totals(user_id, period)
    if not totals:
        return None

    labels = list(totals.keys())
    values = list(totals.values())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 11})
    ax.axis("equal")
    ax.set_title(f"Expenses by category — {period.replace('_', ' ')}")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def set_budget(user_id: str, category: str, monthly_limit: float) -> str:
    if category not in VALID_CATEGORIES:
        category = "other"
    supabase.table("budgets").upsert({
        "user_id": user_id, "category": category, "monthly_limit": monthly_limit
    }, on_conflict="user_id,category").execute()
    return f"Budget set: {category} -> {monthly_limit} EGP/month"


def check_budget_status(user_id: str) -> str:
    budgets = supabase.table("budgets").select("*").eq("user_id", user_id).execute().data
    if not budgets:
        return "No budgets set yet."

    now = datetime.now()
    results = []
    for b in budgets:
        category, limit = b["category"], b["monthly_limit"]
        rows = supabase.table("transactions").select("amount") \
            .eq("user_id", user_id) \
            .gte("txn_date", f"{now.strftime('%Y-%m')}-01") \
            .eq("category", category).eq("type", "expense").execute().data
        spent = sum(r["amount"] for r in rows)
        pct = (spent / limit * 100) if limit > 0 else 0
        status = "OVER BUDGET" if spent > limit else "OK"
        results.append(f"- {category}: {spent:.2f} / {limit:.2f} EGP ({pct:.0f}%) {status}")

    return "\n".join(results)


# =================================================================
# Proactive insights - subscription detection, anomaly alerts,
# and month-end spending projection. These turn the assistant from
# a passive logger into something that notices things on its own.
# =================================================================

def detect_recurring_subscriptions(user_id: str) -> str:
    """Groups expenses by merchant (party) and flags ones that show
    up in at least 2 different calendar months at a similar amount -
    a strong signal of a recurring subscription/bill."""
    rows = supabase.table("transactions").select("*") \
        .eq("user_id", user_id).eq("type", "expense").execute().data

    by_party = {}
    for r in rows:
        party = (r.get("party") or "").strip()
        if not party or party.lower() in ("n/a", "none"):
            continue
        by_party.setdefault(party, []).append(r)

    recurring = []
    for party, txns in by_party.items():
        months_seen = {t["txn_date"][:7] for t in txns}
        if len(months_seen) < 2:
            continue
        amounts = [t["amount"] for t in txns]
        avg = sum(amounts) / len(amounts)
        spread = max(amounts) - min(amounts)
        # tolerant of small variation (e.g. price changes, rounding)
        if spread <= max(avg * 0.15, 15):
            recurring.append((party, avg, len(months_seen)))

    if not recurring:
        return "لسه معنديش بيانات كفاية أرصد بيها اشتراكات متكررة (محتاجين شهرين على الأقل من نفس الجهة)."

    recurring.sort(key=lambda x: -x[1])
    lines = [f"  • {party} - ~{avg:.2f} جنيه/شهر (ظهرت في {months} شهر)" for party, avg, months in recurring]
    total = sum(avg for _, avg, _ in recurring)
    return "الاشتراكات/المدفوعات المتكررة اللي رصدتها:\n" + "\n".join(lines) + \
           f"\n\nإجمالي تقديري شهريًا: {total:.2f} جنيه"


def is_transaction_anomalous(user_id: str, amount: float) -> bool:
    """Flags an expense as unusually large compared to the user's
    recent spending pattern (more than 3x their recent average)."""
    rows = supabase.table("transactions").select("amount") \
        .eq("user_id", user_id).eq("type", "expense") \
        .order("created_at", desc=True).limit(16).execute().data

    baseline = [r["amount"] for r in rows[1:]]  # skip the transaction just inserted
    if len(baseline) < 5:
        return False  # not enough history yet to judge what's "normal"

    avg = sum(baseline) / len(baseline)
    return amount > avg * 3 and amount > 200


def project_month_end_spending(user_id: str) -> str:
    """Extrapolates this month's spending pace to estimate the
    likely total by month-end, based on the daily average so far."""
    now = datetime.now()
    day_of_month = now.day

    if day_of_month < 3:
        return "لسه الشهر بدأ من أيام قليلة، محتاجين بيانات أكتر عشان نطلعلك توقع دقيق."

    totals = get_expense_category_totals(user_id, period="this_month")
    spent_so_far = sum(totals.values())
    if spent_so_far == 0:
        return "مفيش مصاريف مسجلة الشهر ده لحد دلوقتي."

    days_in_month = calendar.monthrange(now.year, now.month)[1]
    daily_rate = spent_so_far / day_of_month
    projected = daily_rate * days_in_month

    return (
        f"صرفت لحد دلوقتي: {spent_so_far:.2f} جنيه في {day_of_month} يوم "
        f"(بمعدل {daily_rate:.2f} جنيه/يوم).\n\n"
        f"لو استمريت بنفس المعدل، المتوقع إجمالي مصاريف الشهر يوصل لحوالي "
        f"{projected:.2f} جنيه."
    )


EGYPTIAN_BANK_SMS_EXAMPLES = """
Real-world examples of Egyptian bank/wallet SMS formats and how to read them
(the exact wording varies by provider, but these patterns are common):

1. "تم خصم مبلغ 250.00 جنيه من حسابك رقم *1234 لصالح كارفور بتاريخ 01-09-2026"
   -> type=expense, amount=250.00, party="كارفور", category=food or shopping

2. "تم سحب مبلغ 1000.00 جنيه من رصيدك عن طريق ماكينة الصراف الآلي ATM"
   -> type=expense, amount=1000.00, party="ATM withdrawal", category=other
   (a raw cash withdrawal - the money left the account, but there's no
   merchant, so don't guess a spending category; use 'other')

3. "تم إيداع مبلغ 15000.00 جنيه في حسابك - مرتب شهر أغسطس"
   -> type=income, amount=15000.00, party="راتب", category=salary

4. "تم تحويل مبلغ 500.00 جنيه من حسابك عبر انستاباي InstaPay إلى محمد أحمد"
   -> type=expense, amount=500.00, party="محمد أحمد", category=transfer
   (InstaPay/mobile transfers OUT of the account are an expense of type 'transfer')

5. "تم استلام تحويل بمبلغ 300.00 جنيه من InstaPay من سارة علي"
   -> type=income, amount=300.00, party="سارة علي", category=transfer

6. "تم خصم 100.00 جنيه من محفظة فودافون كاش الخاصة بك لصالح شحن رصيد"
   -> type=expense, amount=100.00, party="شحن رصيد", category=bills

7. "تم إضافة رصيد بمبلغ 200.00 جنيه إلى محفظة أورانج موني الخاصة بك"
   -> type=income, amount=200.00, party="أورانج موني", category=transfer

8. "عزيزنا العميل، تم خصم 89.99 جنيه اشتراك شهري - نتفليكس"
   -> type=expense, amount=89.99, party="نتفليكس", category=entertainment

Key signal words:
  Expense (money leaving): تم خصم, تم سحب, تم تحويل ... إلى/الى, دفعت, اشتراك
  Income (money arriving): تم إيداع, تم استلام, تم إضافة رصيد, راتب/مرتب, تحويل ... من

Not a transaction at all - reply 'not a transaction', do not call add_transaction:
  OTP / verification codes ("رمز التحقق الخاص بك هو..."), promotional offers,
  balance-check confirmations with no amount changing hands.
"""


TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "add_transaction",
        "description": "Records a new financial transaction (expense or income). "
                        "Extract the amount, category, type, and party from the user's message "
                        "(which is often a raw bank/wallet SMS notification).",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "category": {"type": "string", "enum": VALID_CATEGORIES},
                "type": {"type": "string", "enum": ["expense", "income"]},
                "party": {"type": "string"},
                "raw_text": {"type": "string"},
            },
            "required": ["amount", "category", "type"]
        }
    }},
    {"type": "function", "function": {
        "name": "query_transactions",
        "description": "Retrieves a summary of transactions for a time period, optionally filtered by category.",
        "parameters": {
            "type": "object",
            "properties": {
                "period": {"type": "string", "enum": ["today", "this_month", "last_month", "all"]},
                "category": {"type": "string", "enum": VALID_CATEGORIES},
            },
            "required": ["period"]
        }
    }},
    {"type": "function", "function": {
        "name": "set_budget",
        "description": "Sets a monthly spending limit for a category.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": VALID_CATEGORIES},
                "monthly_limit": {"type": "number"},
            },
            "required": ["category", "monthly_limit"]
        }
    }},
    {"type": "function", "function": {
        "name": "check_budget_status",
        "description": "Shows how much has been spent this month against each set budget.",
        "parameters": {"type": "object", "properties": {}}
    }},
]


def execute_tool(tool_name: str, args: dict, user_id: str) -> str:
    """user_id is injected here, never taken from the LLM's own
    arguments, so a message can never read or write another
    person's data no matter what the model outputs."""
    try:
        if tool_name == "add_transaction":
            args.pop("user_id", None)
            return add_transaction(user_id=user_id, **args)
        elif tool_name == "query_transactions":
            return query_transactions(user_id=user_id, period=args.get("period", "this_month"), category=args.get("category"))
        elif tool_name == "set_budget":
            return set_budget(user_id, args["category"], args["monthly_limit"])
        elif tool_name == "check_budget_status":
            return check_budget_status(user_id)
        else:
            return f"Error: unknown tool '{tool_name}'"
    except Exception as e:
        return f"Error: {e}"


# =================================================================
# Agent Loop - interactive chat version (asks for clarification)
# =================================================================

CHAT_SYSTEM_PROMPT = (
    "You are a personal finance assistant. When the user pastes a raw bank or "
    "e-wallet SMS notification, extract the transaction details and record them "
    "using add_transaction. "
    "IMPORTANT - cash spending: bank/wallet SMS never captures cash transactions "
    "(taxi fare, street food, small purchases), so the user will often just tell "
    "you in plain language instead, e.g. 'دفعت 50 جنيه تاكسي' or 'اشتريت فطار بـ30 "
    "جنيه' or 'paid 100 for groceries in cash'. Treat these exactly like an SMS - "
    "extract the amount, category, and party, and call add_transaction. Don't wait "
    "for a formal bank-style message; any clear statement that money was spent or "
    "received is enough to record, whether typed directly or transcribed from a "
    "voice message (transcriptions may have minor spelling errors - use context "
    "and don't reject a message just because the wording is imperfect). "
    "Carefully decide type='income' (money arriving: deposits, salary, incoming "
    "transfers, refunds) vs type='expense' (money leaving: purchases, debits, "
    "withdrawals, bill payments) based on the wording, including Arabic phrasing "
    "like تم إيداع / استلمت / راتب for income and تم خصم / دفعت / سحب for expense. "
    "For the category: if the merchant name clearly implies one category, use it "
    "directly. But if the merchant is a general store where the purchase could "
    "reasonably be several categories, ask the user to clarify BEFORE recording. "
    "When the user asks about their spending, or asks you to list/show/detail "
    "their recent transactions, use query_transactions - it already returns an "
    "itemized breakdown grouped by category, so just relay it clearly. "
    "Always confirm what you recorded in a short, clear sentence."
    + "\n\n" + EGYPTIAN_BANK_SMS_EXAMPLES
)


def transcribe_voice(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    """Transcribes a voice note to text using Groq's Whisper model, so
    the user can log cash spending by talking instead of typing."""
    buf = io.BytesIO(audio_bytes)
    buf.name = filename
    transcription = groq_client.audio.transcriptions.create(
        file=buf,
        model="whisper-large-v3",
        language="ar",
    )
    return transcription.text


def run_finance_agent(user_message: str, history: list, user_id: str, log_callback=None, max_iterations: int = 5):
    def log(msg):
        if log_callback:
            log_callback(msg)

    if not history:
        history.append({"role": "system", "content": CHAT_SYSTEM_PROMPT})

    history.append({"role": "user", "content": user_message})

    for _ in range(max_iterations):
        response = groq_client.chat.completions.create(
            model=MODEL_NAME, messages=history, tools=TOOLS_SCHEMA, max_tokens=600,
        )
        message = response.choices[0].message
        history.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                log(f"tool: {tool_call.function.name}({args})")
                result = execute_tool(tool_call.function.name, args, user_id)
                log(f"   -> {result[:200]}")
                history.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        else:
            return message.content

    return "Could not complete the request."


# =================================================================
# Webhook mode - single-shot, no back-and-forth possible (an SMS
# forwarder just fires-and-forgets), so it must never ask a
# question - it has to make its best guess and record something.
# =================================================================

WEBHOOK_SYSTEM_PROMPT = (
    "You are a finance assistant processing an automated, one-way SMS forward. "
    "You cannot ask the user anything - always make your best guess and call "
    "add_transaction immediately with a reasonable category. If the message is "
    "not a financial transaction at all (e.g. an OTP code, a promotional SMS), "
    "do not call any tool - just reply 'not a transaction'.\n\n"
    "Deciding expense vs income is critical - read the message carefully:\n"
    "- type='income': money arriving in the user's account. Signals include "
    "words like deposit, received, credited, salary, incoming transfer, refund, "
    "cashback (Arabic: تم إيداع, استلمت, تحويل وارد, راتب/مرتب, تم إضافة رصيد, "
    "استرداد). Use category='salary' for wages, otherwise 'transfer' or 'other'.\n"
    "- type='expense': money leaving the user's account. Signals include debit, "
    "purchase, payment, withdrawal, spent (Arabic: تم خصم, تم الشراء, سحب, "
    "دفعت, خصم من حسابك).\n"
    "When genuinely ambiguous, prefer 'expense' only if there is a debit-like verb; "
    "otherwise still record your best guess rather than skipping it - never leave "
    "a real transaction unrecorded just because you're unsure of the category."
    + "\n\n" + EGYPTIAN_BANK_SMS_EXAMPLES
)


def send_telegram_alert(chat_id: int, text: str) -> None:
    """Best-effort notification - never raises, so a failed alert
    can't itself crash the caller."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text}, timeout=10,
        )
    except Exception:
        pass


def _call_groq_with_retry(messages, max_tokens: int, max_retries: int = 3):
    """Retries transient Groq API failures with exponential backoff
    (1s, 2s, 4s) before giving up. Network hiccups and rate limits
    are common with external APIs and shouldn't silently lose a
    financial transaction on the first blip."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return groq_client.chat.completions.create(
                model=MODEL_NAME, messages=messages, tools=TOOLS_SCHEMA, max_tokens=max_tokens,
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise last_error


def _notify_transaction_recorded(telegram_chat_id: int, args: dict, result: str, user_id: str) -> None:
    """Sends an immediate Telegram confirmation whenever a webhook-forwarded
    SMS gets successfully recorded as a transaction, so the user finds out
    right away instead of only when they next check the bot."""
    amount = args.get("amount", 0)
    category = args.get("category", "other")
    party = args.get("party") or "غير معروف"
    txn_type = args.get("type")

    if txn_type == "income":
        text = f"💰 دخل جديد: +{amount} جنيه\nمن: {party}\nالفئة: {category}"
    else:
        text = f"💸 مصروف جديد: -{amount} جنيه\nلصالح: {party}\nالفئة: {category}"
        if is_transaction_anomalous(user_id, amount):
            text = "⚠️ العملية دي أكبر بكتير من معدل مصاريفك المعتاد - اتأكد إنها صح!\n\n" + text

    send_telegram_alert(telegram_chat_id, text)


def process_incoming_sms(raw_text: str, user_id: str, telegram_chat_id: int = None) -> str:
    messages = [
        {"role": "system", "content": WEBHOOK_SYSTEM_PROMPT},
        {"role": "user", "content": raw_text},
    ]

    for iteration in range(3):
        try:
            response = _call_groq_with_retry(messages, max_tokens=400)
        except Exception as e:
            send_telegram_alert(
                telegram_chat_id,
                "⚠️ وصلتني رسالة SMS بس معرفتش أعالجها بسبب مشكلة مؤقتة في الاتصال.\n"
                "ممكن تبعتها تاني هنا في الشات وأنا هسجلها؟\n\n"
                f"الرسالة: {raw_text[:200]}"
            )
            return f"API error after retries: {e}"

        message = response.choices[0].message
        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    result = f"Error: invalid JSON arguments ({e})"
                else:
                    args["raw_text"] = raw_text
                    result = execute_tool(tool_call.function.name, args, user_id)
                    if tool_call.function.name == "add_transaction" and result.startswith("Recorded:"):
                        _notify_transaction_recorded(telegram_chat_id, args, result, user_id)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        else:
            return message.content

    return "Could not process this message."


# =================================================================
# Monthly summary - used by the scheduled job (see monthly_summary.py)
# =================================================================

SUMMARY_SYSTEM_PROMPT = (
    "You write short, friendly monthly finance summaries in Egyptian Arabic for a "
    "Telegram message. Given raw category totals, write 3-5 sentences: total spent, "
    "the top 1-2 spending categories, total income if any, and one brief, non-judgemental "
    "observation. No headers, no markdown, just plain conversational text."
)


def build_monthly_summary_text(user_id: str) -> str | None:
    """Returns a natural-language monthly summary, or None if the
    user had no transactions last month (skip sending them a message)."""
    raw_data = query_transactions(user_id, period="last_month")
    if raw_data.startswith("No transactions"):
        return None

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": raw_data},
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content