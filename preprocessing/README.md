## Create LLM template

We used a custom template to create a CSV file containing the data for each participant:

        pid,prompt
        64c37e23acd6e500013670a4,"Participant Summary:
        - Q1 (S01: Gender): Weiblich
        - Q2 (S02: Age): 34
        - Q3 (AG: Age Groups): 25-34 Jahre
        - Q4 (S02a: Region): Berlin
        - Q5 (S03: Category usage): In der gesetzlichen Krankenversicherung
        - Q6 (S04: Personal Income): 66.600 € bis 92.999 €
        ...

  We then used a custom template to create prompts suitable for LLMs:
  
          {"prompt": "Eine Person hat folgende Eigenschaften und hat folgende Angaben gemacht. 
          Auf Basis dieser Informationen: Wird diese Person ihre Krankenversicherung in den nächsten 6 Monaten wechseln? 
          Bitte antworte mit \"Nicht-Wechsler ohne Wechselgedanken l10y\", \"Nicht-Wechsler mit Wechselgedanken l10y\" oder \"Wechsler l10y\".
          \n\nSoziodemografie:\nDie Person ist Weiblich, ist 34 Jahre alt, lebt in Berlin, In der gesetzlichen Krankenversicherung versichert, 
          verdient 66.600 € bis 92.999 € ... ", "completion": "Nicht-Wechsler ohne Wechselgedanken l10y"}
