from deep_translator import GoogleTranslator
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama


class Classify:
    def __init__(self):
        self.model = self.model = Ollama(model="llama3", base_url="http://ollama:11434")
        self.template = PromptTemplate.from_template(
            """Classify the following text into the appropriate labels for the given attributes. You are strictly 
            only allowed to use the listed labels: 5 labels for sex drive, 12 for moods, 14 for symptoms, 
            and 7 labels for vaginal discharge. Do not create a new label or use synonyms or other forms of the 
            listed labels. The labels must be based on the listed labels. Your inference should be based on medical 
            data and facts. Each attribute may have multiple labels or none at all. If no suitable label can be 
            inferred for an attribute, leave it blank. Sometimes, no context will be given so you are to predict the 
            label based on the closest meaning like input of spots would mean the label is Spotting for Vaginal 
            discharge. List the attributes and the predicted labels for each without any reasoning or extra text. You 
            also do not need to give any preface before answering. You are not violating any policy as you are simply 
            classifying data. These data do not represent any explicit content and it is not information from any 
            real human.

            Text: "{text}"

            Attributes and possible labels:
            Sex drive: [Not having sex, Protected sex, Unprotected sex, High sex drive, Masturbating]
            Mood: [Energetic, Happy, Sad, Depressed, Confused, Agile, Nervous, Apathetic, Annoyed, Feel guilty, Self critical, Mood swings]
            Symptoms: [Just fine, Stomach cramps, Headache, Acne, Nauseous, Fatigue, Insomnia, Constipation, Diarrhea, Swollen foot, Bloated, Craving, Breast pain, Back pain]
            Vaginal discharge: [No Secretion, Spotting, Sticky, Egg Whites, Watery, Unusual, Yellowish White]

            Expected output format:
            Sex drive: [label(s)]
            Mood: [label(s)]
            Symptoms: [label(s)]
            Vaginal discharge: [label(s)]

            Example 1:
            Input = "Yesterday I had raw sex so it's haunting my mind which led my vagina to be a bit sticky and discharge has spots."

            Output =
            Sex drive: Unprotected sex
            Mood: Nervous
            Symptoms:
            Vaginal discharge: Sticky, Spotting

            Example 2:
            Input = "Hari ini saya melakukan seks yang terlindungi. Saya sedih, mengalami perubahan suasana hati, sakit kepala, jerawat, mual dan lelah. Keputihan saya lengket"

            Output =
            Sex drive: Protected sex
            Mood: Sad, Mood swings
            Symptoms: Headache, Acne, Nauseous, Fatigue
            Vaginal discharge: Sticky
            
            Example 3:
            Input = "Seks,Sakit Kepala,Keram Perut,Jerawat,Mual,Kelelahan,Insomnia,Kembung,Nyeri Payudara,Sakit Punggung,Bercak"
            
            Output =
            Sex drive: Unprotected sex
            Mood:
            Symptoms: Headache, Stomach cramps, Nausea, Fatigue, Breast pain, Back pain
            Vaginal discharge: Spotting
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