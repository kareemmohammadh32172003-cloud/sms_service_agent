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

DUPLICATE_WINDOW_MINUTES = 10


def _is_likely_duplicate(user_id: str, raw_text: str, amount: float) -> bool:
    """Guards against the same SMS being forwarded twice (some SMS
    Forwarder apps retry on flaky connections). Same user, same raw
    text, same amount, within a short time window = duplicate."""
    if not raw_text:
        return False

    cutoff = (datetime.utcnow() - timedelta(minutes=DUPLICATE_WINDOW_MINUTES)).isoformat()
    rows = supabase.table("transactions").select("id") \
        .eq("user_id", user_id) \
        .eq("raw_text", raw_text) \
        .eq("amount", amount) \
        .gte("created_at", cutoff) \
        .execute().data
    return len(rows) > 0


def add_transaction(user_id: str, amount: float, category: str, type: str,
                     party: str = "", raw_text: str = "", txn_date: str = None) -> str:
    if category not in VALID_CATEGORIES:
        category = "other"
    if type not in ("expense", "income"):
        return f"Error: type must be 'expense' or 'income', got '{type}'"

    if _is_likely_duplicate(user_id, raw_text, amount):
        return "Skipped: this looks like a duplicate of a transaction recorded a few minutes ago."

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

    totals = {}
    for r in rows:
        key = (r["category"], r["type"])
        totals.setdefault(key, {"sum": 0.0, "count": 0})
        totals[key]["sum"] += r["amount"]
        totals[key]["count"] += 1

    lines = []
    total_expense = 0.0
    total_income = 0.0
    for (cat, type_), agg in totals.items():
        lines.append(f"- {cat} ({type_}): {agg['sum']:.2f} EGP across {agg['count']} transaction(s)")
        if type_ == "expense":
            total_expense += agg["sum"]
        else:
            total_income += agg["sum"]

    summary = f"Period: {period}\n" + "\n".join(lines)
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
    "Carefully decide type='income' (money arriving: deposits, salary, incoming "
    "transfers, refunds) vs type='expense' (money leaving: purchases, debits, "
    "withdrawals, bill payments) based on the wording, including Arabic phrasing "
    "like تم إيداع / استلمت / راتب for income and تم خصم / دفعت / سحب for expense. "
    "For the category: if the merchant name clearly implies one category, use it "
    "directly. But if the merchant is a general store where the purchase could "
    "reasonably be several categories, ask the user to clarify BEFORE recording. "
    "When the user asks about their spending, use query_transactions. "
    "Always confirm what you recorded in a short, clear sentence."
    + "\n\n" + EGYPTIAN_BANK_SMS_EXAMPLES
)


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


def process_incoming_sms(raw_text: str, user_id: str) -> str:
    messages = [
        {"role": "system", "content": WEBHOOK_SYSTEM_PROMPT},
        {"role": "user", "content": raw_text},
    ]

    for iteration in range(3):
        try:
            response = groq_client.chat.completions.create(
                model=MODEL_NAME, messages=messages, tools=TOOLS_SCHEMA, max_tokens=400,
            )
        except Exception as e:
            return f"API error: {e}"

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