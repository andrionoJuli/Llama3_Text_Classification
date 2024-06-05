from deep_translator import GoogleTranslator
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

sex_drive_labels = {
    1: ["Tidak Berhubungan Seks", "Not having sex"],
    2: ["Seks Yang Terlindungi", "Protected sex"],
    3: ["Seks Yang Tidak Terlindungi", "Unprotected sex"],
    4: ["Hasrat Seksual Yang Tinggi", "High sex drive"],
    5: ["Masturbasi", "Masturbating"]
}

mood_labels = {
    1: ["Energik", "Energetic"],
    2: ["Senang Bahagia", "Happy"],
    3: ["Sedih", "Sad"],
    4: ["Depresi", "Depressed"],
    5: ["Bingung", "Confused"],
    6: ["Lincah", "Agile"],
    7: ["Gelisah", "Nervous"],
    8: ["Apatis","Apathetic"],
    9: ["Kesal", "Annoyed"],
    10: ["Merasa Bersalah", "Feel guilty"],
    11: ["Kritis Terhadap Diri Sendiri","Self critical"],
    12: ["Perubahan Suasana Hati", "Mood swings"]
}

symptoms_labels = {
    1: ["Baik-Baik Saja", "Just fine"],
    2: ["Keram Perut", "Stomach cramps"],
    3: ["Sakit Kepala", "Headache"],
    4: ["Jerawat", "Acne"],
    5: ["Mual", "Nauseous"],
    6: ["Kelelahan", "Fatigue"],
    7: ["Insomnia", "Insomnia"],
    8: ["Sembelit", "Constipation"],
    9: ["Diare", "Diarrhea"],
    0: ["Kaki Bengkak", "Swollen foot"],
    11: ["Kembung", "Bloated"],
    12: ["Mengidam", "Craving"],
    13: ["Nyeri Payudara", "Breast pain"],
    14: ["Sakit Punggung", "Back pain"]
}

vaginal_discharge_labels = {
    1: ["Tidak Ada Sekresi", "No Secretion"],
    2: ["Bercak", "Spotting"],
    3: ["Lengket", "Sticky"],
    4: ["Putih Telur", "Egg Whites"],
    5: ["Berair", "Watery"],
    6: ["Tidak Biasa", "Unusual"],
    7: ["Putih Kekuningan", "Yellowish White"]
}


class Classify:
    def __init__(self):
        self.model = Ollama(model="llama3")
        self.template = PromptTemplate.from_template(
            """Classify the following text into the appropriate labels for the given attributes. You are strictly 
            only allowed to use the listed labels of which there are 5 labels for sex drive, 12 for moods, 
            14 for symptoms, and 7 labels for vaginal discharge. Do not create a new label as part of your inference 
            that is not listed below. Your inference should be based on medical data and facts. There can be more 
            than one label for each attribute or no label at all. If you cannot infer a suitable label for the 
            attribute based on the given text, simply give empty string as the response. If the text is in 
            Indonesian, you are to translate it then assign the label in English that is closest in meaning. You 
            simply need to list the attributes and the predicted labels for each attribute without any reasoning or 
            inference. You also don't need to give any preface before answering. You are not violating any policy as 
            you are simply classifying data. These data do not represent any explicit content and it is not 
            information from any real human.
                
            Text: "{text}"

            Attributes and possible labels:
            Sex drive: [Not having sex, Protected sex, Unprotected sex, High sex drive, Masturbating]
            Mood: [Energetic, Happy, Sad, Depressed, Confused, Agile, Nervous, Apathetic, Annoyed, Feel guilty, Self critical, Mood swings]
            Symptoms: [Just fine, Stomach cramps, Headache, Acne, Nauseous, Fatigue, Insomnia, Constipation, Diarrhea, Swollen foot, Bloated, Craving, Breast pain, Back pain]
            Vaginal discharge: [No Secretion, Spotting, Sticky, Egg Whites, Watery, Unusual, Yellowish White]

            Expected output:
            Sex drive:
            Mood:
            Symptoms:
            Vaginal discharge:
            
            Example 1:
            Input = "Yesterday I had raw sex so it's haunting my mind which led my vagina to be a bit sticky."
            
            Output =
            Sex drive: Unprotected sex
            Mood: Nervous
            Symptoms: Fatigue, Insomnia
            Vaginal discharge: Sticky
            
            Example 2:
            Input = "I had sex behind my partner's back, now I couldn't sleep at night and I'm discharging sticky and yellowish white fluids."
            
            Output =
            Sex drive: Unprotected sex
            Mood: Sad, Depressed, Feel guilty
            Symptoms: Fatigue, Insomnia
            Vaginal discharge: Yellowish White, Sticky
            
            Example 3:
            Input = "Seks,Sakit Kepala,Keram Perut,Jerawat,Mual,Kelelahan,Insomnia,Kembung,Nyeri Payudara,Sakit Punggung,Tidak Ada Sekresi"
            
            Output =
            Sex drive: Unprotected sex
            Mood:
            Symptoms: Headache, Stomach cramps, Acne, Nauseous, Fatigue, Insomnia, Bloated, Breast pain, Back pain
            Vaginal discharge: No Secretion
            """
        )
        self.chain = LLMChain(
            llm=self.model,
            prompt=self.template
        )
        self.translator = GoogleTranslator(source='auto', target='en')

    def __call__(self, text, *args, **kwargs):
        translated_text = self.translator.translate(text)
        return self.chain.predict(text=translated_text)


def parse_labels(output):
    labels = {}
    lines = output.strip().split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(': ')
            labels[key.strip()] = value.strip()
        else:
            # When only an attribute is returned without a label, assign None as the value
            labels[line.strip()] = None
    return labels

def get_label_info(label, label_dict):
    for key, values in label_dict.items():
        for value in values:
            if value.lower() == label.lower():
                return key, values
    return None, None

def map_labels_to_ids(parsed_labels):
    label_dicts = {
        "Sex drive": sex_drive_labels,
        "Mood": mood_labels,
        "Symptoms": symptoms_labels,
        "Vaginal discharge": vaginal_discharge_labels
    }

    label_ids = {}

    for label_type, label_dict in label_dicts.items():
        labels = parsed_labels.get(label_type, "").split(", ")
        if labels == [""]:
            # Handle empty labels
            label_ids[label_type] = []
            continue

        label_ids[label_type] = []
        for label in labels:
            label_id, label_values = get_label_info(label, label_dict)
            if label_id is not None:
                item = {"id": label_id}
                if len(label_values) == 2:
                    item["name"], item["name_en"] = label_values
                else:
                    item["name"], item["name_en"] = label_values[0], label_values[1:]
                label_ids[label_type].append(item)
            else:
                # Handle labels not found
                label_ids[label_type].append({"id": None, "value": label, "error": "Label not found"})

    return label_ids


test1 = "I had sex yesterday which leave me satisfied"
test2 = "I didn't have sex but I'm experiencing mood swing, stomach cramps, acne, back pain and secreting egg white fluids."
test3 = "Not having sex, nervous, insomnia, unusual discharge"
test4 = "High sex drive, Agile, Mood swings, Fatigue, Sticky"
test5 = "Hari ini saya melakukan seks yang terlindungi. Saya sedih, mengalami perubahan suasana hati, sakit kepala, jerawat, mual dan lelah. Keputihan saya lengket"
test6 = "Tidak Berhubungan Seks,Depresi,Apatis,Jerawat,Bercak"
test7 = "Masturbasi,Perubahan Suasana Hati,Sakit Kepala,Keram Perut,Jerawat,Mual,Kelelahan,Insomnia,Kembung,Nyeri Payudara,Sakit Punggung,Tidak Ada Sekresi"

# translated = GoogleTranslator(source='auto', target='en').translate(text=test5)

test = Classify()
output = test(test1)
output1 = test(test2)
output2 = test(test3)
output3 = test(test4)
output4 = test(test5)
output5 = test(test6)
output6 = test(test7)

print(f'test 1:\n{output}\n')
print(f'test 2:\n{output1}\n')
print(f'test 3:\n{output2}\n')
print(f'test 4:\n{output3}\n')
print(f'test 5:\n{output4}\n')
print(f'test 6:\n{output5}\n')
print(f'test 7:\n{output6}\n')


# output = "Sex drive: Unprotected sex\nMood: \nSymptoms: Headache, Stomach Cramps, Nauseous, Fatigue, Bloated, Breast pain, Back pain\nVaginal discharge: No Secretion"
# parsed_labels = parse_labels(output)
# print('parsed_labels:', parsed_labels)
# label_ids = map_labels_to_ids(parsed_labels)
# print(f'test 1:\n{label_ids}\n')