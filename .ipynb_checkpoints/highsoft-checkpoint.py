# Opprett et datasett med informasjon om personens daglige bank og abonnementer
bank_datasett = {
    "Navn": "Sander Hansen",
    "Bankkonto": "1234567890",
    "Saldo": 20000.00,
    "Utskrifter": [
        {
            "Dato": "2023-01-05",
            "Beskrivelse": "Innskudd",
            "Beløp": 1000.00,
        },
        {
            "Dato": "2023-01-10",
            "Beskrivelse": "Netflix-abonnement",
            "Beløp": -120.99,
        },
        {
            "Dato": "2023-01-15",
            "Beskrivelse": "Mobilabonnement",
            "Beløp": -980.00,
        },
        {
            "Dato": "2023-01-20",
            "Beskrivelse": "icloud+",
            "Beløp": -49.99,
        },
        {
            "Dato": "2023-01-25",
            "Beskrivelse": "Lønnsinnskudd",
            "Beløp": 2000.00,
        },
    ]
}

# Funksjon for å finne abonnenter basert på liste med abonnentnavn
def finn_abonnenter(navn_liste, bank_data):
    funnet_abonnenter = []
    for abonnent_navn in navn_liste:
        if abonnent_navn in bank_data["Abonnementer"]:
            funnet_abonnenter.append(abonnent_navn)
    return funnet_abonnenter

# Liste med abonnentnavn du ønsker å finne
ønskede_abonnenter = ["Netflix", "Spotify", "icloud+"]

# Kall funksjonen for å finne abonnenter
funnet_abonnenter = finn_abonnenter(ønskede_abonnenter, bank_datasett)

# Skriv ut resultatet
if funnet_abonnenter:
    print("Følgende abonnenter ble funnet:")
    for abonnent in funnet_abonnenter:
        print(abonnent)
else:
    print("Ingen abonnenter ble funnet blant ønskede abonnentnavn.")
