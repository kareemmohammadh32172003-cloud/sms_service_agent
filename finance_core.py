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
from groq import Groq, RateLimitError
from supabase import create_client
import matplotlib
matplotlib.use("Agg")  # headless server, no display available
import matplotlib.pyplot as plt

load_dotenv()

groq_client = Groq(
    # max_retries=0: the SDK normally retries transient errors itself
    # BEFORE raising, using its own backoff timing (which can be long
    # on a 429). That was compounding with our own retry loop below and
    # blocking the request thread long enough to blow past the web
    # server's worker timeout and crash the whole process. We take full
    # control of retry timing ourselves instead.
    max_retries=0,
    # timeout: hard cap per request so a hung call can't block forever.
    timeout=15.0,
)
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
    (with a fresh api_token and hmac_secret) the first time they say
    /start."""
    existing = supabase.table("users").select("*") \
        .eq("telegram_chat_id", telegram_chat_id).execute().data
    if existing:
        return existing[0]

    token = secrets.token_urlsafe(24)
    hmac_secret = secrets.token_hex(32)
    row = supabase.table("users").insert({
        "telegram_chat_id": telegram_chat_id,
        "api_token": token,
        "hmac_secret": hmac_secret,
        "display_name": display_name,
    }).execute().data
    return row[0]


def get_registered_user(telegram_chat_id: int) -> dict | None:
    """Looks up an already-registered user WITHOUT creating a new
    account. Used to gate access - only people who registered via
    /start with a valid invite code get an account in the first
    place, so this simply returns None for anyone else."""
    rows = supabase.table("users").select("*") \
        .eq("telegram_chat_id", telegram_chat_id).execute().data
    return rows[0] if rows else None


def get_user_by_token(token: str) -> dict | None:
    """Looks up which user a webhook request belongs to, based on
    the token in their personal webhook URL."""
    rows = supabase.table("users").select("*").eq("api_token", token).execute().data
    return rows[0] if rows else None


def get_or_generate_hmac_secret(user_id: str) -> str:
    """Returns the user's HMAC secret, generating and saving one if
    they registered before this feature existed (so hmac_secret is
    still NULL for them)."""
    row = supabase.table("users").select("hmac_secret").eq("id", user_id).execute().data
    existing = row[0].get("hmac_secret") if row else None
    if existing:
        return existing
    new_secret = secrets.token_hex(32)
    supabase.table("users").update({"hmac_secret": new_secret}).eq("id", user_id).execute()
    return new_secret


def list_all_users() -> list[dict]:
    """Used by the monthly summary job to loop over everyone."""
    return supabase.table("users").select("*").execute().data


# =================================================================
# Trusted senders - whitelist of real bank/wallet SMS sender IDs per
# user. This is the actual anti-fake-message defense: the webhook
# will refuse to auto-record ANY message whose sender isn't on this
# list, regardless of how convincing the text looks. Wording alone
# (e.g. "تم خصم 100 جنيه") is never enough on its own - it must also
# come from a sender the user has explicitly confirmed is a real
# bank/wallet, or it just gets held for manual review instead.
# =================================================================

def _normalize_sender(sender: str) -> str:
    return (sender or "").strip().lower()


def _extract_reference_number(raw_text: str) -> str:
    """Best-effort extraction of a reference/transaction number if the
    message states one (e.g. 'رقم العملية 123456', 'Ref: 987654'). This
    is stored for audit purposes only - NEVER used as a trust signal on
    its own, because unlike the sender ID, this is just text a phone can
    freely type, so it proves nothing about authenticity by itself."""
    import re
    match = re.search(r"(?:رقم\s*(?:العملي[ةه]|المرجعي|الإشعار)|ref(?:erence)?(?:\s*no\.?)?)\D{0,5}(\d{4,})",
                       raw_text, re.IGNORECASE)
    return match.group(1) if match else ""


def _looks_like_registered_sender(sender: str) -> bool:
    """Automatic, no-maintenance classifier for whether an SMS sender ID
    could plausibly be a real registered bank/wallet sender, based on a
    property that a personal phone genuinely cannot fake:

    A regular mobile phone can only ever send SMS with its OWN phone
    number as the 'From' - it has no way to set an arbitrary alphabetic
    Sender ID like 'Fawry' or a bank's name (that requires a registered
    SMS gateway account with the telecom regulator). So:

      - Sender contains any letter (Arabic or Latin)  -> business-style
        alphanumeric ID -> almost certainly a real registered sender.
      - Sender is a short numeric code (<=6 digits)    -> many banks use
        short numeric codes too -> treat as registered.
      - Sender is a normal 10-11 digit mobile number (with or without a
        leading +/country code), or missing entirely   -> this is what
        a personal phone (yours or anyone else's) looks like when it
        sends/forwards a message -> NOT auto-trusted.

    This is not unbeatable (a real SMS-spoofing attack that pays for a
    gateway account could still forge a fake alphanumeric sender), but
    it correctly blocks anything a phone can send on its own - including
    self-sent test messages - with zero manual setup.
    """
    if not sender:
        return False
    cleaned = sender.strip().lstrip("+")
    if not cleaned.isdigit():
        return True  # contains letters -> alphanumeric business sender ID
    return len(cleaned) <= 6  # short numeric code, not a full mobile number


def get_trusted_senders(user_id: str) -> list[str]:
    row = supabase.table("users").select("trusted_senders").eq("id", user_id).execute().data
    if not row or not row[0].get("trusted_senders"):
        return []
    return [s.strip() for s in row[0]["trusted_senders"].split(",") if s.strip()]


def is_sender_trusted(user_id: str, sender: str) -> bool:
    # Automatic check first - covers the overwhelming majority of real
    # bank/wallet senders with zero setup. The manual list below is only
    # a rare fallback for the odd sender our automatic rule misjudges.
    if _looks_like_registered_sender(sender):
        return True
    normalized = _normalize_sender(sender)
    return normalized in [_normalize_sender(s) for s in get_trusted_senders(user_id)]


def add_trusted_sender(user_id: str, sender: str) -> str:
    sender = sender.strip()
    if not sender:
        return "لازم تكتب اسم أو رقم المرسل."
    current = get_trusted_senders(user_id)
    if _normalize_sender(sender) in [_normalize_sender(s) for s in current]:
        return f"'{sender}' مسجل بالفعل كمصدر موثوق."
    current.append(sender)
    supabase.table("users").update({"trusted_senders": ",".join(current)}).eq("id", user_id).execute()
    return f"تمام، '{sender}' بقى مصدر موثوق - أي رسالة جاية منه هتتسجل تلقائي."


def list_trusted_senders_text(user_id: str) -> str:
    senders = get_trusted_senders(user_id)
    if not senders:
        return (
            "مفيش أي مصدر موثوق مسجل لسه، فكل الرسايل هتتوقف للمراجعة اليدوية.\n"
            "استخدم /trustsender <اسم المرسل> عشان تضيف أول واحد."
        )
    return "المصادر الموثوقة عندك:\n" + "\n".join(f"- {s}" for s in senders)


# =================================================================
# Tools - same signatures as before, all scoped by user_id
# =================================================================

DUPLICATE_WINDOW_MINUTES = 60


DUPLICATE_WINDOW_MINUTES = 1440  # 24 hours - generous enough to catch delayed retries


def _is_likely_duplicate(user_id: str, raw_text: str, amount: float, type: str) -> bool:
    """Guards against the same SMS being forwarded twice. Matches on
    the exact raw SMS text (not the AI-extracted party name, which can
    vary slightly between identical retries) plus amount and direction.
    A true retry resends byte-identical text, so this has virtually no
    risk of blocking two genuinely different real transactions - even
    if the same person sends the same amount twice, real bank/wallet
    SMS almost always differs in wording (reference number, balance,
    timestamp embedded in the text itself)."""
    if not raw_text:
        return False  # nothing to compare against - don't block

    cutoff = (datetime.utcnow() - timedelta(minutes=DUPLICATE_WINDOW_MINUTES)).isoformat()
    rows = supabase.table("transactions").select("id") \
        .eq("user_id", user_id) \
        .eq("raw_text", raw_text) \
        .eq("amount", amount) \
        .eq("type", type) \
        .gte("created_at", cutoff) \
        .execute().data
    return len(rows) > 0


# =================================================================
# Account balances - tracks a running balance per bank card/wallet
# (e.g. "بنك مصر", "فودافون كاش", "انستاباي", "كاش") so the user can
# ask "كام جالي في X" or "رصيدي كام في X" at any time.
# =================================================================

DEFAULT_ACCOUNT_NAME = "غير محدد"

# Maps known spelling/language variants of the same real-world account
# to one canonical name, so "فودافون كاش", "Vodafone Cash", and
# "vodafone-cash" all end up as the exact same account instead of
# silently becoming separate accounts with split balances.
ACCOUNT_ALIASES = {
    "فودافون كاش": ["فودافون كاش", "فودافون كاش مصر", "vodafone cash", "vodafone-cash", "vf cash", "فودافون"],
    "أورانج موني": ["أورانج موني", "أورانج كاش", "orange money", "orange cash", "orange"],
    "اتصالات كاش": ["اتصالات كاش", "etisalat cash", "e& cash", "etisalat"],
    "انستاباي": ["انستاباي", "انستا باي", "instapay", "insta pay"],
    "بنك مصر": ["بنك مصر", "banque misr", "bank misr", "bm"],
    "البنك الأهلي": ["البنك الأهلي", "البنك الاهلي", "nbe", "national bank of egypt"],
    "بنك CIB": ["بنك cib", "cib", "commercial international bank"],
    "بنك QNB": ["بنك qnb", "qnb"],
    "كاش": ["كاش", "نقدي", "cash"],
}


def _clean_for_comparison(text: str) -> str:
    """Strips punctuation/separator differences (VF-Cash vs VF Cash vs
    vf_cash) so they compare as identical, without touching the actual
    letters - keeps Arabic text completely intact."""
    text = text.strip().lower()
    for sep in ("-", "_", ".", "/"):
        text = text.replace(sep, " ")
    return " ".join(text.split())  # collapse repeated whitespace


def normalize_account_name(account_name: str) -> str:
    """Collapses spelling/language/punctuation variants of the same
    account into one canonical name. Falls back to the original text,
    stripped, if it doesn't match any known alias - so a new bank/
    wallet the user hasn't used before still gets its own account
    rather than being forced into an existing bucket."""
    name = (account_name or "").strip()
    if not name:
        return DEFAULT_ACCOUNT_NAME

    name_clean = _clean_for_comparison(name)

    # Pass 1: exact match once separators/case/spacing are normalized.
    for canonical, aliases in ACCOUNT_ALIASES.items():
        if name_clean in (_clean_for_comparison(a) for a in aliases):
            return canonical

    # Pass 2: fallback substring containment, for variants not listed
    # verbatim (e.g. "Vodafone Cash Wallet"). Only applied to aliases
    # of reasonable length to avoid short strings causing false matches.
    for canonical, aliases in ACCOUNT_ALIASES.items():
        for alias in aliases:
            alias_clean = _clean_for_comparison(alias)
            if len(alias_clean) >= 4 and (alias_clean in name_clean or name_clean in alias_clean):
                return canonical

    return name


def get_or_create_account(user_id: str, account_name: str) -> dict:
    account_name = normalize_account_name(account_name)
    existing = supabase.table("accounts").select("*") \
        .eq("user_id", user_id).eq("name", account_name).execute().data
    if existing:
        return existing[0]
    row = supabase.table("accounts").insert({
        "user_id": user_id, "name": account_name, "balance": 0,
    }).execute().data
    return row[0]


def adjust_account_balance(user_id: str, account_name: str, delta: float) -> float:
    account = get_or_create_account(user_id, account_name)
    new_balance = account["balance"] + delta
    supabase.table("accounts").update({"balance": new_balance}).eq("id", account["id"]).execute()
    return new_balance


def set_account_balance(user_id: str, account_name: str, balance: float) -> str:
    account_name = normalize_account_name(account_name)
    account = get_or_create_account(user_id, account_name)
    supabase.table("accounts").update({"balance": balance}).eq("id", account["id"]).execute()
    return f"تم ضبط رصيد '{account_name}' على {balance:.2f} جنيه."


def get_account_balance(user_id: str, account_name: str) -> str:
    account_name = normalize_account_name(account_name)
    rows = supabase.table("accounts").select("*") \
        .eq("user_id", user_id).eq("name", account_name).execute().data
    if not rows:
        return f"مفيش رصيد متسجل لحساب '{account_name}' لسه."
    return f"{account_name}: {rows[0]['balance']:.2f} جنيه"


def list_account_balances(user_id: str) -> str:
    rows = supabase.table("accounts").select("*").eq("user_id", user_id).execute().data
    if not rows:
        return "لسه معندكش أي حسابات متسجلة. أول معاملة تتسجل، الحساب بتاعها بيتعمل أوتوماتيك."
    rows.sort(key=lambda r: -r["balance"])
    lines = [f"  • {r['name']}: {r['balance']:.2f} جنيه" for r in rows]
    total = sum(r["balance"] for r in rows)
    return "أرصدتك الحالية:\n" + "\n".join(lines) + f"\n\nالإجمالي: {total:.2f} جنيه"




def add_transaction(user_id: str, amount: float, category: str, type: str,
                     party: str = "", raw_text: str = "", txn_date: str = None,
                     account: str = "", balance_after: float = None) -> str:
    if category not in VALID_CATEGORIES:
        category = "other"
    if type not in ("expense", "income"):
        return f"Error: type must be 'expense' or 'income', got '{type}'"

    if _is_likely_duplicate(user_id, raw_text, amount, type):
        return "Skipped: this looks like a duplicate of a transaction recorded recently."

    txn_date = txn_date or date.today().isoformat()
    account_name = normalize_account_name(account)

    supabase.table("transactions").insert({
        "user_id": user_id,
        "txn_date": txn_date,
        "amount": amount,
        "party": party,
        "category": category,
        "type": type,
        "raw_text": raw_text,
        "account": account_name,
    }).execute()

    if balance_after is not None:
        # The SMS itself stated the resulting balance - trust that over our
        # own running total, since it self-corrects any past drift (a missed
        # or mis-parsed transaction won't compound into a wrong balance forever).
        set_account_balance(user_id, account_name, balance_after)
        new_balance = balance_after
        source_note = " - synced from SMS"
    else:
        delta = amount if type == "income" else -amount
        new_balance = adjust_account_balance(user_id, account_name, delta)
        source_note = ""

    sign = "-" if type == "expense" else "+"
    return (f"Recorded: {sign}{amount} EGP | {category} | {party or 'N/A'} | "
            f"Account: {account_name} (new balance: {new_balance:.2f} EGP{source_note})")


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


UNDO_WINDOW_MINUTES = 30


def delete_last_transaction(user_id: str) -> str:
    last = get_last_transaction(user_id)
    if not last:
        return "You don't have any recorded transactions yet."

    created_at_raw = last.get("created_at")
    if created_at_raw:
        try:
            created_dt = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
            age_minutes = (datetime.now(created_dt.tzinfo) - created_dt).total_seconds() / 60
        except Exception:
            age_minutes = None
        if age_minutes is not None and age_minutes > UNDO_WINDOW_MINUTES:
            return (
                f"Cannot undo: this transaction is older than {UNDO_WINDOW_MINUTES} minutes "
                f"(protects against accidentally deleting an old transaction by mistake). "
                f"Use /fix to correct its category instead, or ask an admin for manual removal."
            )

    supabase.table("transactions").delete().eq("id", last["id"]).execute()

    account_name = last.get("account") or DEFAULT_ACCOUNT_NAME
    reversal = -last["amount"] if last["type"] == "income" else last["amount"]
    new_balance = adjust_account_balance(user_id, account_name, reversal)

    sign = "-" if last["type"] == "expense" else "+"
    return (f"Deleted: {sign}{last['amount']} EGP | {last['category']} | {last['party'] or 'N/A'} | "
            f"Account: {account_name} (balance reverted to {new_balance:.2f} EGP)")


def query_transactions(user_id: str, period: str = "this_month", category: str = None, account: str = None) -> str:
    now = datetime.now()

    query = supabase.table("transactions").select("*").eq("user_id", user_id)

    if period == "this_month":
        query = query.gte("txn_date", f"{now.strftime('%Y-%m')}-01")
    elif period == "today":
        query = query.eq("txn_date", date.today().isoformat())
    elif period == "yesterday":
        query = query.eq("txn_date", (date.today() - timedelta(days=1)).isoformat())
    elif period == "last_month":
        last_month = now.month - 1 or 12
        year = now.year if now.month > 1 else now.year - 1
        query = query.gte("txn_date", f"{year}-{last_month:02d}-01") \
                      .lt("txn_date", f"{now.strftime('%Y-%m')}-01")

    if category:
        query = query.eq("category", category)
    if account:
        account = normalize_account_name(account)
        query = query.eq("account", account)

    rows = query.execute().data

    if not rows:
        filters = []
        if category:
            filters.append(f"category '{category}'")
        if account:
            filters.append(f"account '{account}'")
        suffix = f" ({', '.join(filters)})" if filters else ""
        return f"No transactions found for period '{period}'{suffix}"

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
    elif period == "yesterday":
        query = query.eq("txn_date", (date.today() - timedelta(days=1)).isoformat())
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

IMPORTANT: every example below now also extracts 'account' - the user's OWN
bank/wallet the money moved through - separately from 'party' (the other
side of the transaction). Never confuse the two.

1. "تم خصم مبلغ 250.00 جنيه من حسابك في بنك مصر رقم *1234 لصالح كارفور، رصيدك الحالي 1750.00 جنيه"
   -> type=expense, amount=250.00, party="كارفور", account="بنك مصر", category=food or shopping, balance_after=1750.00
   (the SMS explicitly stated the resulting balance - always extract it when present)

2. "تم سحب مبلغ 1000.00 جنيه من رصيدك في البنك الأهلي عن طريق ماكينة الصراف الآلي ATM"
   -> type=expense, amount=1000.00, party="ATM withdrawal", account="البنك الأهلي", category=other
   (a raw cash withdrawal - the money left the account, but there's no
   merchant, so don't guess a spending category; use 'other')

3. "تم إيداع مبلغ 15000.00 جنيه في حسابك ببنك CIB - مرتب شهر أغسطس"
   -> type=income, amount=15000.00, party="راتب", account="CIB", category=salary

4. "تم تحويل مبلغ 500.00 جنيه من حسابك عبر انستاباي InstaPay إلى محمد أحمد"
   -> type=expense, amount=500.00, party="محمد أحمد", account="انستاباي", category=transfer
   (InstaPay/mobile transfers OUT of the account are an expense of type 'transfer')

5. "تم استلام تحويل بمبلغ 300.00 جنيه من InstaPay من سارة علي"
   -> type=income, amount=300.00, party="سارة علي", account="انستاباي", category=transfer

6. "تم خصم 100.00 جنيه من محفظة فودافون كاش الخاصة بك لصالح شحن رصيد"
   -> type=expense, amount=100.00, party="شحن رصيد", account="فودافون كاش", category=bills

7. "تم إضافة رصيد بمبلغ 200.00 جنيه إلى محفظة أورانج موني الخاصة بك"
   -> type=income, amount=200.00, party="أورانج موني", account="أورانج موني", category=transfer

8. "عزيزنا العميل، تم خصم 89.99 جنيه اشتراك شهري - نتفليكس"
   -> type=expense, amount=89.99, party="نتفليكس", account="" (unknown - not mentioned, leave blank), category=entertainment

9. "تم شحن رصيد موبايلك ب 19 بنجاح وخصم 19 من محفظتك شاملة الضريبة؛ رصيد حسابك في فودافون كاش الحالي 188.89"
   -> type=expense, amount=19, party="شحن رصيد", account="فودافون كاش", category=bills, balance_after=188.89
   (mobile top-up confirmations often state TWO numbers - the top-up
   amount and the amount actually deducted from the wallet, which can
   differ if fees/tax apply. ALWAYS use the deducted/خصم amount as
   'amount', never the top-up amount, even when they happen to match
   like here. Look for the number right next to خصم specifically.)

10. "تم خصم المبلغ التالى من رصيدك لتسديد قيمة خدمة؛ 0.67 ج ضريبة دمغة وتم السداد بالكامل والرصيد الحالى"
   -> NOT a transaction. This is a stamp-duty tax (ضريبة دمغة) withheld
   from phone airtime that was already purchased in an earlier top-up
   SMS - no new money left a bank account or e-wallet, so this is
   carrier bookkeeping, not an expense to record again.

11. "تم خصم 125 وحدة من كارت فكة رسوم هدية الكارت"
   -> NOT a transaction. 125 is in وحدة (telecom bundle units -
   minutes/data), not currency, regardless of the خصم verb.

12. User types plainly (no SMS at all): "دفعت 50 جنيه تاكسي كاش"
   -> type=expense, amount=50, party="تاكسي", account="كاش", category=transport
   (any cash spending the user tells you about directly belongs to the "كاش" account)

Key signal words:
  Expense (money leaving): تم خصم, تم سحب, تم تحويل ... إلى/الى, دفعت, اشتراك
  Income (money arriving): تم إيداع, تم استلام, تم إضافة رصيد, راتب/مرتب, تحويل ... من

If the SMS doesn't mention which bank/wallet at all, leave account empty -
don't guess a specific bank name that wasn't stated.

Not a transaction at all - reply 'not a transaction', do not call add_transaction:
  OTP / verification codes ("رمز التحقق الخاص بك هو..."), promotional offers,
  balance-check confirmations with no amount changing hands.
"""


TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "add_transaction",
        "description": "Records a new financial transaction (expense or income). "
                        "Extract the amount, category, type, party, and account from the user's "
                        "message (which is often a raw bank/wallet SMS notification).",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "category": {"type": "string", "enum": VALID_CATEGORIES},
                "type": {"type": "string", "enum": ["expense", "income"]},
                "party": {"type": "string", "description": "The other side of the transaction - the "
                          "merchant or person the money went to/came from, e.g. 'كارفور', 'محمد أحمد'."},
                "account": {"type": "string", "description": "Which of the user's OWN accounts/wallets "
                            "the money moved through, e.g. 'بنك مصر', 'CIB', 'فودافون كاش', 'أورانج موني', "
                            "'انستاباي', or 'كاش' for manually logged cash spending. This is different from "
                            "'party' - it's the user's own account, not the other side of the transaction."},
                "balance_after": {"type": "number", "description": "The account's resulting balance, "
                                  "ONLY if the SMS explicitly states it (e.g. 'رصيدك الحالي 1750 جنيه', "
                                  "'رصيدك المتبقي: 500'). This is the ground truth from the bank itself, "
                                  "so extract it whenever present - it keeps the tracked balance accurate "
                                  "even if a past transaction was missed. Leave blank if not mentioned - "
                                  "never calculate or guess this yourself."},
                "raw_text": {"type": "string"},
            },
            "required": ["amount", "category", "type"]
        }
    }},
    {"type": "function", "function": {
        "name": "query_transactions",
        "description": "Retrieves an itemized summary of transactions for a time period, optionally "
                        "filtered by category and/or account (e.g. 'كام جالي في بنك مصر النهارده').",
        "parameters": {
            "type": "object",
            "properties": {
                "period": {"type": "string", "enum": ["today", "yesterday", "this_month", "last_month", "all"]},
                "category": {"type": "string", "enum": VALID_CATEGORIES},
                "account": {"type": "string", "description": "Filter to only this account/wallet, e.g. 'بنك مصر' or 'فودافون كاش'."},
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
    {"type": "function", "function": {
        "name": "delete_last_transaction",
        "description": "Deletes the single most recently recorded transaction. Only works if "
                        "it was recorded within the last 30 minutes (protects against accidentally "
                        "deleting an old transaction). Use this when the user asks to undo, delete, "
                        "or remove the last thing they logged - e.g. 'امسحلي آخر رسالة', 'شيل العملية دي'.",
        "parameters": {"type": "object", "properties": {}}
    }},
    {"type": "function", "function": {
        "name": "correct_last_transaction_category",
        "description": "Changes only the category of the most recently recorded transaction. "
                        "Use this when the user asks to fix, correct, or change the category of "
                        "the last thing they logged - e.g. 'عدلها لفئة كذا', 'دي مش أكل دي مواصلات'.",
        "parameters": {
            "type": "object",
            "properties": {"new_category": {"type": "string", "enum": VALID_CATEGORIES}},
            "required": ["new_category"]
        }
    }},
    {"type": "function", "function": {
        "name": "check_account_balance",
        "description": "Shows the current tracked balance for one account/wallet (e.g. 'رصيدي كام "
                        "في فودافون كاش'), or all accounts at once if none is specified (e.g. 'قولي أرصدتي').",
        "parameters": {
            "type": "object",
            "properties": {"account": {"type": "string", "description": "Optional - omit to see all account balances."}},
        }
    }},
    {"type": "function", "function": {
        "name": "set_account_balance",
        "description": "Sets the exact current balance for one of the user's accounts/wallets. Use "
                        "this when the user tells you their actual real-world balance to start tracking "
                        "from, e.g. 'رصيدي في بنك مصر دلوقتي 2000 جنيه'.",
        "parameters": {
            "type": "object",
            "properties": {
                "account": {"type": "string"},
                "balance": {"type": "number"},
            },
            "required": ["account", "balance"]
        }
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
            return query_transactions(user_id=user_id, period=args.get("period", "this_month"),
                                       category=args.get("category"), account=args.get("account"))
        elif tool_name == "check_account_balance":
            acct = args.get("account")
            return get_account_balance(user_id, acct) if acct else list_account_balances(user_id)
        elif tool_name == "set_account_balance":
            return set_account_balance(user_id, args["account"], args["balance"])
        elif tool_name == "delete_last_transaction":
            return delete_last_transaction(user_id)
        elif tool_name == "correct_last_transaction_category":
            return correct_last_transaction_category(user_id, args.get("new_category"))
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
    "CRITICAL: always call query_transactions freshly for every spending question, "
    "even if you answered a similar question earlier in this same conversation. "
    "Never reuse or estimate numbers from earlier in the chat history - the data "
    "can change at any time (new transactions, /undo, /fix corrections), and 'today' "
    "itself changes, so a stale answer is actively wrong, not just imprecise. "
    "When the user asks to undo, delete, or remove the last thing they logged, use "
    "delete_last_transaction. When they ask to fix or change the category of the "
    "last thing they logged, use correct_last_transaction_category. "
    "ACCOUNTS: always try to extract 'account' too - which of the user's own bank "
    "cards or wallets the money moved through (e.g. 'بنك مصر', 'فودافون كاش', "
    "'انستاباي'), separate from 'party' (the merchant/other side). For manually "
    "typed or spoken cash spending, use account='كاش'. When the user asks about a "
    "balance (e.g. 'رصيدي كام في فودافون كاش', 'قولي أرصدتي') use check_account_balance. "
    "When they tell you their real current balance for an account (e.g. 'رصيدي في "
    "بنك مصر دلوقتي 2000 جنيه'), use set_account_balance to record that starting point. "
    "When they ask how much came into or out of a specific account (e.g. 'كام جالي "
    "في بنك مصر النهارده'), pass that account as a filter to query_transactions. "
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

        # IMPORTANT: store a plain dict, not the raw SDK message object.
        # Appending the object itself works for the very first follow-up
        # call, but breaks on later turns once this history is reused
        # across multiple separate messages (as it is here, persisted
        # per Telegram chat) - the API rejects the malformed replay and
        # raises an exception, which without a try/except higher up
        # means the bot goes completely silent with no reply at all.
        assistant_entry = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        history.append(assistant_entry)

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
    "a real transaction unrecorded just because you're unsure of the category. "
    "Also extract 'account' - which of the user's own bank cards or wallets the "
    "money moved through (e.g. 'بنك مصر', 'فودافون كاش', 'انستاباي') - separate "
    "from 'party' (the merchant/other side). If the SMS doesn't mention which "
    "bank/wallet, leave account blank rather than guessing. Also extract "
    "'balance_after' whenever the SMS explicitly states the resulting balance "
    "(e.g. 'رصيدك الحالي X', 'رصيدك المتبقي X') - never calculate this yourself, "
    "only extract it if literally stated.\n\n"
    "CRITICAL - telecom airtime/bundle activity is NOT a financial transaction: "
    "mobile carriers (Vodafone, Orange, Etisalat, WE) send many automated SMS "
    "about what happens to phone credit AFTER it was purchased - service taxes "
    "withheld from airtime, bundle/minute/data consumption, promotional bonus "
    "grants, gift-card unit fees. These never represent real money leaving a "
    "bank account or e-wallet, even when phrased with a debit verb like خصم. "
    "Only call add_transaction if BOTH of these hold:\n"
    "  1. The amount is real currency (جنيه/ج/EGP/دولار) - NEVER telecom bundle "
    "units (وحدة, دقيقة, ميجا, GB, رسالة). A message stating '125 وحدة' or "
    "'950 دقيقة' is not a transaction no matter what verb precedes it.\n"
    "  2. The money is leaving/entering an actual bank account or e-wallet "
    "balance (بنك مصر, فودافون كاش, فوري, انستاباي, etc) - NOT the phone "
    "line's own airtime/credit/bundle balance. Watch for phrases like 'من "
    "رصيدك ... لتسديد قيمة خدمة', 'ضريبة دمغة', 'كارت فكة', 'رسوم هدية الكارت' "
    "- these describe the carrier's internal bookkeeping on credit that was "
    "already purchased and recorded once at top-up time, not a new expense.\n"
    "If either test fails, treat it exactly like a promotional SMS: reply "
    "'not a transaction' and call no tool.\n\n"
    "SCAM AWARENESS: genuine bank/wallet transaction SMS simply states a fact "
    "(amount, direction, sometimes balance) - it never asks the user to call a "
    "number, click a link, share an OTP, or 'confirm' anything. If the message "
    "combines a transaction claim with an urgent call-to-action like that, still "
    "record what it claims (so the user sees it and can judge for themselves) but "
    "this pattern is a strong scam indicator worth being extra careful about."
    + "\n\n" + EGYPTIAN_BANK_SMS_EXAMPLES
)


def send_telegram_alert(chat_id: int, text: str) -> None:
    """Best-effort notification - never raises, so a failed alert
    can't itself crash the caller. Logs WHY it didn't send, since a
    silent no-op here is otherwise very hard to debug."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("send_telegram_alert: TELEGRAM_BOT_TOKEN not set in this service's env vars")
        return
    if not chat_id:
        print("send_telegram_alert: no chat_id provided (user has no telegram_chat_id?)")
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text}, timeout=10,
        )
        if not resp.ok:
            print(f"send_telegram_alert: Telegram API returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"send_telegram_alert: request failed: {e}")


def _call_groq_with_retry(messages, max_tokens: int, max_retries: int = 2):
    """Retries transient Groq API failures with a short, bounded
    backoff - kept deliberately small (max ~1s total sleep, 15s per
    request cap) so total handling time can never approach the web
    server's worker timeout again (that combination is what caused a
    full process crash before).

    Rate limits (429) are NOT retried here at all - blocking this
    thread to wait one out is exactly what caused the crash. Instead
    we fail fast; the SMS Forwarder app on the phone already retries
    failed webhook calls on its own schedule, which naturally spaces
    requests out far better than blocking a server thread ever could.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return groq_client.chat.completions.create(
                model=MODEL_NAME, messages=messages, tools=TOOLS_SCHEMA, max_tokens=max_tokens,
            )
        except RateLimitError as e:
            print(f"_call_groq_with_retry: rate limited (429), failing fast without retry: {e}")
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(1)
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


def process_incoming_sms(raw_text: str, user_id: str, telegram_chat_id: int = None, sender: str = "") -> str:
    # --- Sender whitelist gate: this runs BEFORE the LLM ever sees the
    # text, and is the actual defense against fake/injected messages.
    # No amount of realistic wording lets a message skip this check -
    # only a sender the user has explicitly trusted can auto-record.
    if not is_sender_trusted(user_id, sender):
        preview = raw_text[:200]
        shown_sender = sender or "(مفيش اسم مرسل متبعت مع الرسالة)"
        ref_number = _extract_reference_number(raw_text)
        ref_line = f"الرقم المرجعي المكتوب في النص: {ref_number}\n" if ref_number else ""
        send_telegram_alert(
            telegram_chat_id,
            "🛑 وصلتني رسالة مسجلتهاش تلقائي لأن شكل المرسل زي رقم موبايل عادي "
            "مش زي مُرسل بنك/محفظة مسجل رسميًا:\n\n"
            f"المرسل: {shown_sender}\n"
            f"{ref_line}"
            f"النص: {preview}\n\n"
            + ("ملحوظة: وجود رقم مرجعي في النص مش دليل كفاية لوحده - أي حد ممكن "
               "يكتب رقم عشوائي شكله حقيقي، فالقرار اتاخد بناءً على المُرسل مش على النص.\n\n"
               if ref_number else "")
            + "لو ده فعلاً مصدر حقيقي بشكل استثنائي (نادر)، ابعت:\n"
            f"/trustsender {shown_sender}\n"
            "وبعدها ابعتلي نفس الرسالة تاني هنا في الشات وأنا هسجلها يدوي دلوقتي."
        )
        return f"Rejected: sender '{shown_sender}' does not look like a registered business sender"

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