"""Synthetic generator of French bank transaction labels."""

import copy
import random
import re

import pandas as pd
from faker import Faker
from src.preprocessing import strip_accents

fake = Faker("fr_FR")

CITIES = [
    "PARIS", "PARIS 11", "PARIS 15", "LYON", "LYON 3", "MARSEILLE", "TOULOUSE",
    "NICE", "NANTES", "STRASBOURG", "MONTPELLIER", "BORDEAUX", "LILLE", "RENNES",
    "REIMS", "SAINT-DENIS", "BOULOGNE-BILL", "MONTREUIL", "NANTERRE", "VERSAILLES",
    "CRETEIL", "GRENOBLE", "DIJON", "ANGERS", "NIMES", "AIX-EN-PROVEN", "ROUEN",
    "TOULON", "LE HAVRE", "VILLEURBANNE", "ARGENTEUIL", "LA DEFENSE",
]

LEGAL_FORMS = ["SAS", "SARL", "SA", "EURL", "SNC"]

BASE_CATALOG = {
    "Groceries": {
        "card": [
            "CARREFOUR", "CARREFOUR MARKET", "CARREFOUR CITY", "LIDL", "ALDI",
            "AUCHAN", "E.LECLERC", "LECLERC DRIVE", "MONOPRIX", "FRANPRIX",
            "INTERMARCHE", "SUPER U", "PICARD", "BIOCOOP", "NATURALIA",
            "GRAND FRAIS", "CASINO SHOP", "G20", "LA VIE CLAIRE",
        ],
    },
    "Restaurants": {
        "card": [
            "MCDONALDS", "BURGER KING", "KFC", "QUICK", "STARBUCKS", "COLUMBUS CAFE",
            "PAUL", "BRIOCHE DOREE", "DOMINOS PIZZA", "SUSHI SHOP", "O TACOS",
            "FIVE GUYS", "BUFFALO GRILL", "HIPPOPOTAMUS", "DELIVEROO", "UBER EATS",
            "JUST EAT", "SUMUP *CAFE DES AMIS", "ZETTLE *FOOD TRUCK",
            "BRASSERIE DU MARCHE", "CAFE DE LA GARE",
        ],
    },
    "Transport": {
        "card": [
            "SNCF CONNECT", "SNCF INTERNET", "OUIGO", "RATP", "UBER *TRIP", "BOLT.EU",
            "HEETCH", "BLABLACAR", "LIME", "DOTT", "TOTALENERGIES", "ESSO", "SHELL",
            "BP", "AVIA", "INDIGO PARKING", "SAEMES", "SANEF", "VINCI AUTOROUTES",
        ],
        "debit": ["NAVIGO", "COMUTITRES", "VELIB METROPOLE"],
    },
    "Subscriptions": {
        "card": [
            "NETFLIX.COM", "SPOTIFY", "DISNEY PLUS", "AMAZON PRIME", "APPLE.COM/BILL",
            "YOUTUBE PREMIUM", "GOOGLE *GOOGLE ONE", "MICROSOFT*365",
            "OPENAI *CHATGPT", "ADOBE", "AUDIBLE",
        ],
        "debit": ["CANAL+", "DEEZER", "LE MONDE", "L EQUIPE", "DAZN", "BEIN SPORTS"],
    },
    "Shopping": {
        "card": [
            "AMAZON", "AMAZON MKTPLACE", "FNAC", "DARTY", "BOULANGER", "DECATHLON",
            "ZARA", "H&M", "UNIQLO", "KIABI", "PRIMARK", "IKEA", "LEROY MERLIN",
            "CASTORAMA", "SEPHORA", "NOCIBE", "ACTION", "CDISCOUNT", "ZALANDO",
            "VINTED", "LEBONCOIN", "APPLE STORE", "GIFI",
        ],
    },
    "Housing & Utilities": {
        "debit": [
            "EDF", "ENGIE", "TOTALENERGIES ELECTRICITE GAZ", "FREE MOBILE",
            "FREE TELECOM", "ORANGE", "SFR", "BOUYGUES TELECOM", "SOSH", "RED BY SFR",
            "VEOLIA EAU", "SUEZ EAU", "MAIF", "MACIF", "AXA FRANCE",
            "DGFIP TAXE FONCIERE", "FONCIA",
        ],
        "transfer_out": ["LOYER", "AGENCE FONCIA", "NEXITY", "SCI LES LILAS", "CITYA"],
    },
    "Health": {
        "card": [
            "PHARMACIE DU CENTRE", "PHARMACIE DE LA GARE", "PHARMACIE LAFAYETTE",
            "GRANDE PHARMACIE", "DOCTOLIB*DR LEROY", "CABINET DENTAIRE",
            "CERBALLIANCE", "OPTIC 2000", "AFFLELOU", "CABINET KINE",
        ],
        "debit": ["MUTUELLE HARMONIE", "MGEN", "ALAN", "MALAKOFF HUMANIS"],
    },
    "Leisure & Travel": {
        "card": [
            "UGC", "PATHE", "MK2", "FNAC SPECTACLES", "TICKETMASTER", "AIRBNB",
            "BOOKING.COM", "AIR FRANCE", "EASYJET", "RYANAIR", "TRANSAVIA", "STEAM",
            "PLAYSTATION NETWORK", "NINTENDO", "DISNEYLAND PARIS", "PARC ASTERIX",
            "MUSEE DU LOUVRE",
        ],
        "debit": ["BASIC FIT", "FITNESS PARK", "NEONESS", "ON AIR FITNESS"],
    },
    "Income": {
        "transfer_in": [
            "CAF", "CAF DE PARIS", "CPAM", "CPAM PARIS", "FRANCE TRAVAIL",
            "DGFIP REMBOURSEMENT IMPOT",
        ],
    },
    "Transfers": {
        "transfer_out": ["LYDIA", "LIVRET A", "COMPTE EPARGNE", "WISE", "REVOLUT"],
        "transfer_in": ["LYDIA"],
    },
    "Cash Withdrawal": {
        "withdrawal": [
            "BNP PARIBAS", "SOCIETE GENERALE", "CREDIT AGRICOLE", "LCL", "CIC",
            "CREDIT MUTUEL", "LA BANQUE POSTALE", "CAISSE D EPARGNE",
            "BANQUE POPULAIRE", "HSBC",
        ],
    },
    "Bank Fees": {
        "fee": [
            "FRAIS TENUE DE COMPTE", "COTISATION CARTE VISA", "COTIS CARTE GOLD MC",
            "COMMISSION D INTERVENTION", "FRAIS REJET PRLV", "AGIOS",
            "INTERETS DEBITEURS", "FRAIS PAIEMENT HORS ZONE EURO",
            "COTIS OFFRE GROUPEE", "FRAIS ENVOI CHEQUIER",
        ],
    },
}


# ---------- Noise helpers ----------

def merchant_key(name):
    """Normalize a merchant name for comparison: 'G 20', 'G20' and 'g-20' give the same key."""
    return re.sub(r"[^A-Z0-9]", "", strip_accents(name.upper()))


def random_date():
    return fake.date_between(start_date="-1y", end_date="today").strftime("%d/%m")


def random_city():
    return random.choice(CITIES)


def random_card():
    return f"CARTE X{fake.random_number(digits=4, fix_len=True)}"


def random_ref():
    return fake.bothify("FR##ZZZ######")


# ---------- Label builders (one per payment type) ----------

def card_label(merchant):
    date, city, card = random_date(), random_city(), random_card()
    templates = [
        f"CB {merchant} {date}",
        f"CB {merchant} {city} {date}",
        f"CARTE {date} {merchant} {city}",
        f"CB {merchant} {date} {card}",
        f"{card} {date} {merchant}",
        f"PAIEMENT PAR CARTE {merchant} {date}",
        f"PAIEMENT CB {date} {city} {merchant}",
    ]
    return random.choice(templates)


def direct_debit_label(merchant):
    ref = random_ref()
    templates = [
        f"PRLV SEPA {merchant}",
        f"PRLV SEPA {merchant} {ref}",
        f"PRELEVEMENT {merchant} REF {ref}",
        f"PRLV {merchant} {random_date()}",
    ]
    return random.choice(templates)


def transfer_out_label(recipient):
    templates = [
        f"VIR SEPA EMIS VERS {recipient}",
        f"VIR INST VERS {recipient}",
        f"VIREMENT EN FAVEUR DE {recipient}",
        f"VIR {recipient} {random_date()}",
    ]
    return random.choice(templates)


def transfer_in_label(sender):
    templates = [
        f"VIR SEPA RECU DE {sender}",
        f"VIR INST RECU {sender}",
        f"VIREMENT DE {sender}",
        f"VIR {sender} REF {random_ref()}",
    ]
    return random.choice(templates)


def withdrawal_label(bank):
    date, city = random_date(), random_city()
    templates = [
        f"RETRAIT DAB {date} {city}",
        f"RETRAIT DAB {bank} {city} {date}",
        f"RET DAB {date} {city} {random_card()}",
        f"RETRAIT {bank} {date}",
    ]
    return random.choice(templates)


def fee_label(fee):
    month = fake.date_between(start_date="-1y", end_date="today").strftime("%m/%Y")
    templates = [fee, f"{fee} {random_date()}", f"{fee} {month}"]
    return random.choice(templates)


LABEL_BUILDERS = {
    "card": card_label,
    "debit": direct_debit_label,
    "transfer_out": transfer_out_label,
    "transfer_in": transfer_in_label,
    "withdrawal": withdrawal_label,
    "fee": fee_label,
}


# ---------- Catalog and dataset ----------

def build_catalog(extra_merchants=None):
    """Return the full catalog: base merchants, generated names, and optional extra merchants."""
    catalog = copy.deepcopy(BASE_CATALOG)

    employers = [
        strip_accents(f"{fake.last_name()} {random.choice(LEGAL_FORMS)}".upper())
        for _ in range(15)
    ]
    people = [strip_accents(f"{fake.last_name()} {fake.first_name()}".upper()) for _ in range(40)]
    catalog["Income"]["transfer_in"] += employers
    catalog["Transfers"]["transfer_out"] += people[:25]
    catalog["Transfers"]["transfer_in"] += people[25:]

    if extra_merchants:
        known_keys = {
            merchant_key(merchant)
            for groups in catalog.values()
            for merchants in groups.values()
            for merchant in merchants
        }
        for category, groups in extra_merchants.items():
            for payment_type, merchants in groups.items():
                existing = catalog[category].setdefault(payment_type, [])
                for merchant in merchants:
                    merchant = strip_accents(merchant.upper()).strip()
                    key = merchant_key(merchant)
                    if key and key not in known_keys:
                        existing.append(merchant)
                        known_keys.add(key)

    return catalog


def generate_dataset(n_per_category=500, extra_merchants=None, seed=42):
    """Generate a labeled, deduplicated DataFrame of synthetic transaction labels."""
    random.seed(seed)
    Faker.seed(seed)
    catalog = build_catalog(extra_merchants)

    rows = []
    for category, groups in catalog.items():
        for _ in range(n_per_category):
            payment_type = random.choice(list(groups))
            merchant = random.choice(groups[payment_type])
            rows.append({
                "label": LABEL_BUILDERS[payment_type](merchant),
                "category": category,
                "merchant": merchant,
                "payment_type": payment_type,
            })

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset="label").reset_index(drop=True)