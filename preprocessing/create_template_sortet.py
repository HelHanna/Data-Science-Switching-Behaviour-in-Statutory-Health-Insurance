import pandas as pd
import re
import json

def extract_field(text, label):
    # Regex sucht nach dem Label mit ": <Wert>" danach
    match = re.search(rf"- {label}\):\s*(.+)", text)
    if match:
        value = match.group(1).strip()
        if value.lower() not in ["keine angabe", "weiß ich nicht", "nan", ""]:
            return value
    return None


FIELDS_GROUPED = {
    "Soziodemografie": [
    (r"Q1 \(S01: Gender", "Die Person ist {}"),
    (r"Q2 \(S02: Age", "ist {} Jahre alt"),
    (r"Q4 \(S02a: Region", "lebt in {}"),
    (r"Q5 \(S03: Category usage", "{} versichert"),
    (r"Q6 \(S04: Personal Income", "verdient {}"),
    ],
    
    "Haushalt & Beruf": [
    (r"Q98.*", "Haushaltsgröße: {} Personen"),
    (r"Q99.*", "hat {}"),
    (r"Q100.*", "Migrationshintergrund: {}"),
    (r"Q101.*", "aus: {}"),
    (r"Q102.*", "arbeitet: {}"),
    (r"Q103.*", "Berufserfahrung: {}"),
    (r"Q104.*", "Beziehungsstatus: {}"),
    ],

    "Markenbekanntheit: Die Person kann folgende Krankenkassen ohne Hinweis nennen": [
    (r"Q7.1.1 \(Q7.1.1", "{}"),
    (r"Q7.1.2 \(Q7.1.2", "{}"),
    (r"Q7.1.3 \(Q7.1.3", "{}"),
    (r"Q7.1.4 \(Q7.1.4", "{}"),
    (r"Q7.1.5 \(Q7.1.5", "{}"),
    (r"Q7.1.6 \(Q7.1.6", "{}"),
    (r"Q8 \(S06: Aided Awareness", "Die Person kennt folgende Krankenkassen, wenn sie genannt werden: {}"),
    ],

    "Nutzung & Wechsel": [
    (r"Q9 \(S07: Usage", "ist bei der {} versichert"),
    (r"Q10 \(S07a: Usage detail", "genauer bei {} versichert"),
    #(r"Q11 \(S07_dummy", "bei {} versichert"),
    (r"Q12 \(S08: Time of consideration change", "hat über einen Wechsel {} nachgedacht"),
    (r"Q14 \(S10: Number of changes", "hat {} die Versicherung gewechselt"),
    (r"Q15 \(S11: Change from PKV to GKV", "hat von der PKV zur GKV gewechselt: {}"),
    (r"Q16 \(S15: Planned change n6m", "{}"),
    ],

    "Wissen / Selbstwahrnehmung": [
    (r"Q19 \(G01: GKV Know-How Self-Assessment", "schätzt ihr Wissen über GKV als {} ein"),
    (r"Q20\.1", "sagt {}, Wenn die Krankenkasse den Zusatzbeitrag erhöht, haben Versicherte ein Sonderkündigungsrecht"),
    (r"Q20\.2", "sagt {}, Wenn der Zusatzbeitrag erhöht wird, wird die Differenz vom Arbeitgeber getragen"),
    (r"Q20\.3", "sagt {}, Jede Krankenkasse darf neben dem regulären Beitragssatz auch noch einen Zusatzbeitrag erheben"),
    (r"Q20\.4", "sagt {}, Meine aktuelle Krankenkasse hat in diesem Jahr ihren Zusatzbeitrag erhöht"),
    (r"Q20\.5", "sagt {}, Man kann seine Krankenkasse ohne Angabe von Gründen in einer gesetzlich festgelegten Kündigungsfrist kündigen"),
    (r"Q20\.6", "sagt {}, Gesetzliche Krankenkassen dürfen jemanden aufgrund von Vorerkrankungen ablehnen"),
    (r"Q20\.7", "sagt {}, Man kann selbst die Krankenkasse wählen, bei der man versichert sein will"),
    ],

    "Wechselgründe – allgemein & konkret: Die Person gibt folgendes als Wechselgründe an": [
    (r"Q24 \(R00: Moment of truth", "{}"),
    (r"Q25 \(R01_TextDummy", "Die Person antwortet auf folgende Frage: {}"),
    (r"Q26 \(R01: Moment of truth", "mit: {}"),
    (r"Q27 \(R02_TextDummy", "Die Person antwortet auf folgende Frage: {}"),
    (r"Q28 \(R02: Moment of truth in detail", "mit: {}"),
    ],

    "Zufriedenheit / Leistungsbewertung": [
    (r"Q40.1.*", "Die Person bewertet die Aussage: Sehr gutes Angebot eines Bonusprogramms z.B. Zusätzliche Gesundheits-Check-Ups, Auszahlungen, Sportprogramme etc als '{}' bei der eigenen Krankenkasse"),
    (r"Q40.2.*", "bewertet die Aussage :Hervorragendes Angebot zusätzlicher Leistungen (z.B. Zahnreinigung, Osteopathie, Kurse und Programme zur Gesundheitsvorsorge wie Rückenschule, Raucherentwöhnung oder Gewichtsreduktion, etc als '{}' bei der eigenen Krankenkasse"),
    (r"Q40.3.*", "bewertet die Aussage: Größtes Angebot an weiteren Zusatzversicherungen z.B. Zahnzusatz, Auslandskrankenschutz etc als '{}' bei der eigenen Krankenkasse"),
    (r"Q40.4.*", "bewertet die Aussage:Tarife mit Beitragsrückerstattungen z.B. Wahltarif oder Gutscheinen für Zusatzleistungen bei gesundheitsbewusstem Verhalten als '{}' bei der eigenen Krankenkasse"),
    (r"Q40.5.*", "bewertet die Aussage: Besondere Versorgungsangebote z.B. besondere Versorgung für spezifische Erkrankungen, Arztterminvermittlung, Zweitmeinung, telemedizinische Versorgung, etc. als '{}' bei der eigenen Krankenkasse"),
    (r"Q40.6.*", "bewertet die Aussage: Vorreiter in Sachen Innovation bspw. Telemedizin als '{}' bei der eigenen Krankenkasse"),
    (r"Q40.7.*", "bewertet die Aussage: Exzellente Beratung zu verschiedenen Gesundheitsthemen (z.B. zu Leistungen, Vorsorgeuntersuchungen, Krankheiten und Präventionsmaßnahmen als '{}' bei der eigenen Krankenkasse"),
    (r"Q40.8.*", "bewertet die Aussage: Umfangreicher digitaler Kundenservice z.B. Live-/Videochat, Apps, Chatbot, papierlose Dokumentenübermittlung, etc. als '{}' bei der eigenen Krankenkasse"),
    (r"Q40.9.*", "bewertet die Aussage: Uneingeschränkte Erreichbarkeit des Kundenservice z.B. Öffnungszeiten, Geschäftsstelle vor Ort als '{}' bei der eigenen Krankenkasse"),
    (r"Q40.10.*", "bewertet die Aussage: Äußerst vertrauensvoller und unterstützender Kundenumgang als '{}' bei der eigenen Krankenkasse"),
    (r"Q40.11.*", "bewertet die Aussage: Schnellstmögliche Bearbeitung von Anliegen als '{}' bei der eigenen Krankenkasse"),
    (r"Q40.12.*", "bewertet die Aussage: Niedriger Preis (Geringer Zusatzbeitrag im Vergleich zu anderen Krankenkassen) als '{}' beid der eigenen Krankenkasse"),
    (r"Q40.13.*", "bewertet die Aussage: Hervorragender Ruf der Krankenkasse als '{}' bei der eigenen Krankenkasse"),
    (r"Q40.14.*", "bewertet die Aussage: Nachweisliches Engagement der Krankenkasse zum Thema Nachhaltigkeit als '{}' bei der eigenen Krankenkasse"),
    ],
    
    "Loyalität / Empfehlung": [
    (r"Q41 \(M03: GKV Loyalty", "Die Person fühlt sich ihrer GKV {} "),
    (r"Q42 \(M04a: NPS", "auf einer Skala von 1 bis 10 würde ihre Krankenkasse mit '{}/10' weiterempfehlen"),
    (r"Q44 \(M04b: NPS - Reasoning", "Begründung für Weiterempfehlung: {}"),
    ],

    "Informationsverhalten & Abschluss": [
    (r"Q47 \(C01a: Information channels.*", "Die Person informiert sich über {}"),
    (r"Q49 \(C01b: Information in detail", "die ausschlaggebende Informationsquelle ist: {}"),
    (r"Q52 \(C02: Purchase channel", "der bevorzugte Abschlusskanal ist: {}"),
    ],

    "Wechselbarrieren": [
    (r"Q56.1.*", "Die Person fühlt sich auf einer Skala von 1 bis 13 '{}/13' verbunden mit der Krankenkasse"),
    (r"Q56.2.*", "Die Person hat auf einer Skala von 1 bis 13 '{}/13' Angst vor Verlust von Vorteilen als langjähriger Versicherter"),
    (r"Q56.3.*", "Die Person bewertet auf einer Skala von 1 bis 13 die Komplexität des Wechselprozesses als '{}/13'"),
    (r"Q56.4.*", "Die Person bewertet auf einer Skala von 1 bis 13 den zeitlichen Aufwand für den Vergleich von Krankenkassen als '{}/13'"),
    (r"Q56.5.*", "Die Person bewertet auf einer Skala von 1 bis 13 die Unklarheit über zusätzliche Kosten für Leistungen bei den Krankenkassen als '{}/13'"),
    (r"Q56.6.*", "Die Person bewertet auf einer Skala von 1 bis 13 Mangelnde persönliche Beratungsmöglichkeiten als '{}/13'"),
    (r"Q56.7.*", "Die Person bewertet auf einer Skala von 1 bis 13 die Einwände von Freunden/Familien als '{}/13'"),
    (r"Q56.8.*", "Die Person bewertet auf einer Skala von 1 bis 13 die Schwierigkeiten, das Leistungsangebot der Krankenkassen zu vergleichen als '{}/13'"),
    (r"Q56.9.*", "Die Person bewertet auf einer Skala von 1 bis 13 die Befürchtung, vorübergehend keinen Krankenversicherungsschutz zu haben als '{}/13'"),
    (r"Q56.10.*", "Die Person bewertet auf einer Skala von 1 bis 13 die Angst vor schlechteren Leistungen nach einem Wechsel als '{}/13'"),
    (r"Q56.11.*", "auf einer Skala von 1 bis 13, bewertet die Person die Aussage: Es ist mir nicht wichtig, bei welcher Krankenkasse ich versichert bin als '{}/13'"),
    (r"Q56.12.*", "auf einer Skala von 1 bis 13, bewertet die Person die Aussage: Mich hat keine andere Krankenkasse überzeugt als '{}/13'"),
    (r"Q56.13.*", "auf einer Skala von 1 bis 13, bewertet die Person die Aussage: Bürokratischer Aufwand für den Kassenwechsel zu hoch als '{}/13'"),
    ],
    
    "Preisakzeptanz": [
    (r"Q58.*", "hält einen Wechsel bei einem Zusatzbeitrag für {}"),
    ],

    "Kontaktverhalten": [
    (r"Q69.1.*", "Online Kontakt mit GKV: {}"),
    (r"Q69.2.*", "telefonischer Kontakt mit GKV: {}"),
    (r"Q69.3.*", "Kontakt in Geschäftsstelle: {}"),
    (r"Q69.4.*", "Briefkontakt mit GKV: {}"),
    (r"Q70.*", "bevorzugter Kontaktweg mit der GKV: {}"),
    (r"Q71.*", "bevorzugter Kontaktaufnahme von der GKV: {}"),
    (r"Q72.*", "letzter Kontakt war: {}"),
    (r"Q73.*", "kontaktierte die GKV wegen {}"),
    (r"Q74.*", "bewertet den Kontakt mit der GKV als {}"),
    (r"Q75.*", "positives Kontaktbeispiel: {}"),
    (r"Q76.*", "negatives Kontakbeispiel: {}"),
    ],

    "Zusatzversicherungen & weitere Anbieter": [
    (r"Q77.*", "Die Person hat folgende Zusatzversicherungen: {}"),
    (r"Q78.*", "Zusatzversicherung Anbieter: {}"),
    ],
    
    "Andere Verträge:": [
    (r"Q79.1.*", "Strom-/Gasanbieter: {}"),
    (r"Q79.2.*", "Mobilfunkanbieter: {}"),
    (r"Q79.3.*", "Internetanbieter: {}"),
    (r"Q79.4.*", "Kfz-Versicherung: {}"),
    ],

    "PKV Wechsel": [
    (r"Q80.*", "Gedanken zum Wechsel in die PKV: {}"),
    (r"Q83.*", "Wechselgründe: {}"),
    (r"Q84.*", "Wechselinteresse zur PKV: {}"),
    (r"Q85.*", "Situationen für einen Wechsel wären: {}"),
    (r"Q86.*", "{}"),
    (r"Q87.*", "ausgewählte Wechselgründe: {}"),
    (r"Q88.1.*", "auf einer Skala von 1 bis 9, bewertet die Person günstigere Beiträge als in gesetzlicher Krankenkasse mit '{}' als Wechselgrund"),
    (r"Q88.2.*", "auf einer Skala von 1 bis 9, bewertet die Person bessere und zusätzliche Versicherungsleistungen mit '{}' als Wechselgrund bewertet"),
    (r"Q88.3.*", "auf einer Skala von 1 bis 9, bewertet die Person eine bessere Behandlung im Krankenhaus (Chefarztbehandlung, Einzelzimmer, etc. mit '{}' als Wechselgrund"),
    (r"Q88.4.*", "auf einer Skala von 1 bis 9, bewertet die Person eine größere Flexibilität bei Leistungen und Behandlungen mit '{}' als Wechselgrund"),
    (r"Q88.5.*", "auf einer Skala von 1 bis 9, bewertet die Person schnellere Termine/kürzere Wartezeiten bei Ärzten mit '{}' als Wechselgrund"),
    (r"Q88.6.*", "auf einer Skala von 1 bis 9, bewertet die Person den Zugang zu Privatärzten mit '{}' als Wechselgrund"),
    (r"Q88.7.*", "auf einer Skala von 1 bis 9, bewertet die Person höhere Servicequalität und Komfort (z.B. kürzere Wartezeiten am Telefon) mit '{}' als Wechselgrund"),
    (r"Q88.8.*", "auf einer Skala von 1 bis 9, bewertet die Person einen höheren Versichertenstatus/Prestige mit '{}' als Wechselgrund"),
    (r"Q89.*", "andere Wechselgründe sind: {}"),
    (r"Q90.*", "PKV-Informationskanal: {}"),
    (r"Q91.*", "PKV-Abschlusskanal: {}"),
    ],

    "Gesundheit & Prävention": [
    (r"Q93.*", "Die Person trifft folgende Gesundheitsvorsorge: {}"),
    (r"Q94.*", "hat folgende Erkrankungen: {}"),
    (r"Q95.*", "weitere Erkrankungen: {}"),
    ],

    
}

# Fixed prompt instruction
prompt_intro = (
    "Eine Person hat folgende Eigenschaften und hat folgende Angaben gemacht. "
    "Auf Basis dieser Informationen: Wird diese Person ihre Krankenversicherung in den nächsten 6 Monaten wechseln? "
    "Bitte antworte mit \"Nicht-Wechsler ohne Wechselgedanken l10y\", \"Nicht-Wechsler mit Wechselgedanken l10y\" oder \"Wechsler l10y\"."
)


def generate_summary_from_text(text):
    prompt_parts = [prompt_intro]

    for group_name, fields in FIELDS_GROUPED.items():
        group_sentences = []
        for pattern, template in fields:
            value = extract_field(text, pattern)
            if value:
                try:
                    group_sentences.append(template.format(value))
                except Exception:
                    group_sentences.append(f"{template} {value}")
        if group_sentences:
            section = f"{group_name}:\n" + ", ".join(group_sentences) + "."
            prompt_parts.append(section)

    return "\n\n".join(prompt_parts)





# CSV laden
df = pd.read_csv("participant_prompts.csv")
#df = pd.read_csv("C:/Users/hanna/OneDrive - Universität des Saarlandes/Dokumente/Semester_10/data science/preprocessing/participant_prompts.csv")

# JSONL schreiben
# with open("participant_prompts.jsonl", "w", encoding="utf-8") as f:
#     for _, row in df.iterrows():
#         summary = generate_summary_from_text(row['prompt'])  # <- passt hier, weil die Spalte prompt heißt
#         f.write(json.dumps({"prompt": summary}, ensure_ascii=False) + "\n")


with open("participant_prompts.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        summary = generate_summary_from_text(row['prompt'])
        #completion = extract_field(row['prompt'], r"Q18 \(S17: Final Wording Dummy (hidden variable)")
        completion = extract_field(row['prompt'], r"Q18 \(S17.*")

        
        if completion:  # only write rows where Q19 is present and valid
            f.write(json.dumps({
                "prompt": summary,
                "completion": completion
            }, ensure_ascii=False) + "\n")
