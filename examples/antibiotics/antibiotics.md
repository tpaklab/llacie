(Drug Name: str: The name of the drug, 1 to 2 words long
        Route: str | The route with which the medication is given
        Dose: str | Dose given in mg or g
        Frequency: str | Daily/weekly etc.
        Duration: int | Duration in days
        Start date: str | MM/DD/YYYY format
        End date: str | MM/DD/YYYY format)
Take a look the files in this folder for existing notes.
Add to the csv file until there are 50 total notes (in batches of 5).
For each note_id in the csv file, create a realistic clinical note based on it.
Ensure there is some missing information based on the frequency of that information in real clinical notes.
In general the drug should always be present, use your judgement for the rest. 
Ensure the notes have some missing info like a missing start date / end date etc etc. This should also be reflected in the .csv file that creates the labels where it must be marked empty.
You can have multiple drugs per clinical note, but you are not required to. 
Ensure the drugs show the diversity in real-world cases, not extremely niche, but include some rarer ones too. 