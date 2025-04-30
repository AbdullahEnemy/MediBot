import json
import os 
import logging
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
import httpx
# from openai import OpenAI
import requests
from typing import Optional, Dict
from pydantic import BaseModel, Field, field_validator
from langchain_core.output_parsers import JsonOutputParser
import io
from fastapi import  File, UploadFile
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv




logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s:%(message)s:%(funcName)s')
file_handler = logging.FileHandler('gppod_medicalGPT.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
load_dotenv()


class Texttospeech(BaseModel):
    text: str
    id: int

class Report(BaseModel):
    chat_id: int
    patientId: str


class ConversationQuery(BaseModel):
    chat_id: Optional[int] = None
    Id: int
    patientId: str
    user_query: str
    name: Optional[str] = None
    age : Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    firstMessage: bool
    language_code: str

class AddTitle(BaseModel):
    chatId: int
    name: str


conv_stages_summary_dict = {
1: "1 - 'presenting complaint' is the conversation stage 1, i.e 1st stage, In this stage, Ask the patient about their primary symptoms and focus on identifying primary symptoms. After gathering primary symptoms, move to second stage 'Complaint History' , otherwise stay in current stage. Make the conversation personalized based on information collected (Name, Age, Gender, Occupation) Exit Criteria : Patient has provided primary symptoms.",
2: "2 - 'complaint history' is the conversation stage 2, i.e. second 2nd stage, which comes after 'presenting complaint' stage. In this stage, Based on the primary symptoms of the patient, Gather detailed information on primary symptoms' duration, progression, and associated symptoms. After collecting duration, progression, and associated symptoms, proceed to 'Patient History' stage. If Primary symptoms not available, Move to first stage 'Final Diagnosis'. Exit Criteria : Patient has provided duration, progression, and associated symptoms.",
3: "3 - 'final diagnosis' is the conversation stage 5, i.e. third 3rd stage, which comes after 'complaint history' stage. In this stage, make a diagnosis of patient condition. Before diagnosing, consider the patient responses of previous conversation stages and medical history, and exclude improbable conditions to narrow down possibilities. Summarize findings and propose one or two likely diagnoses with reasoning. Exit Criteria : You have provided one diagnoses with reasoning",
4: "4 - 'treatment plan' is the conversation stage 6, i.e. forth 4th stage, which comes after 'final diagnosis' stage. In this stage, Suggest a treatment startegy to patient for which may consist of lifestyle changes, Home remedies or UK-specific medications. Check for drug allergies before prescibing any medicine. Do not refer patients to specialists, schedule appointments, or suggest further medical evaluation. Exit Criteria : You have suggested a treatment strategy to patient.",
5: "5 - 'closure' is the conversation stage 7, i.e. fifth 5th stage, which comes after 'treatment plan' stage. In this stage, Close the conversation in a professional and polite manner."}




conv_stages_summary_str = "\n".join([v for _,v in conv_stages_summary_dict.items()])

    
def Openai_Models_List(openai_api_key):
    try:
        response = httpx.get(f"https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {openai_api_key}"})
        models = response.json()
        model_list = []
        for item in models['data']:
            model_list.append(item['id'])
        return model_list
    except Exception as ex:
        return Exception(f"Error in fetching OpenAI models: {ex}")
    
async def create_conv_stage_and_history_pair(history, stage_analyzer_chain,  chat_id = None):
    conv_stage_map = {} 
    chain_output = await stage_analyzer_chain.ainvoke({'conversation_history': history, 
                                                    'conv_stages_summary':conv_stages_summary_str})
    chain_output = chain_output.content
    stage_num = chain_output[0]
    conv_stage_map[stage_num] = history

    return conv_stage_map

class ConversationStageAnalyzer(Runnable):
    @classmethod
    def from_openai_llm(cls, llm_name:str) -> Runnable: 
        try:
            openai_api_key = os.getenv('OPENAI_API_KEY')
            assert openai_api_key is not None, "Please set the OPENAI_API_KEY environment variable"
            
            openai_models_list = Openai_Models_List(openai_api_key)
            assert llm_name in openai_models_list, f"Model {llm_name} not found in OpenAI Models List"
        
        except AssertionError as ex:
            return Exception(f"Given model {llm_name} not found in OpenAI Models List")
        except Exception as ex:
            return Exception(f"Error in fetching OpenAI models: {ex}") 
        
        try:
            stage_analyzer_prompt_str = """
        Analyze the conversation history enclosed between the markers '===' to determine the next immediate conversation stage for a patient healthcare conversation. Don't move to next stage unless the all the information asked in previous stage is provided, do not ask so many questions in one response 

        ===
        conversation history : {conversation_history}
        ===

        Based on the conversation history, choose the next appropriate conversation stage from the following conversation stages:
        conversation stages: {conv_stages_summary}. 

        Based on your choice of next appropriate conversation stage, provide a SINGLE DIGIT response between 1 to 8.
        Your Response MUST always start with a SINGLE DIGIT between 1 to 8 , representing your best guess for the next appropriate conversation stage.

        Instructions for Response Generation:
        - Your response MUST always start with SINGLE DIGIT Integer, without any additional text.
        - If there is no conversation history provided, respond with 1.
        - You can provide Explanation or additional text in your response. But it must be after the SINGLE DIGIT Integer.

        Examples of Valid Responses:
        Example1 : 3
        Example2 : 4  \n Explanation : ....
        Example3 : 2  \n\n Explanation

        Examples of Invalid Responses:
        Example1 : Three
        Example2 : 4th Stage
        Example3 : Stage 5
        Example4 : Next stage is 6
        Example4 : Response : 4
        Example5 : This response indicates that the next appropriate conversation stage is 'Complaint History'

        Response : 
        """
            stage_analyzer_prompt = ChatPromptTemplate.from_template(template=stage_analyzer_prompt_str)
            chat_llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=os.environ.get('OPENAI_API_KEY'))
            stage_analyzer_chain = ( stage_analyzer_prompt 
                         | chat_llm )
            return stage_analyzer_chain  
        except Exception as ex:
            raise Exception(f"Error in creating ConversationStageAnalyzer: {ex}")


class MedicalConversationChain(Runnable):


    @classmethod
    def from_openai_llm(cls, llm_name:str ='gpt-4o-mini') -> Runnable: 
        try:
            openai_api_key = os.getenv('OPENAI_API_KEY')
            assert openai_api_key is not None, "Please set the OPENAI_API_KEY environment variable"
            
            openai_models_list = Openai_Models_List(openai_api_key)
            assert llm_name in openai_models_list, f"Model {llm_name} not found in OpenAI Models List"
        
        except AssertionError as ex:
            return Exception(f"Given model {llm_name} not found in OpenAI Models List")
        except Exception as ex:
            return Exception(f"Error in fetching OpenAI models: {ex}") 
        
        try:
            physician_agent_prompt_str = (
    """
    You are "Dr. {physician_name}," an AI medical assistant designed to support a General Physician.
    You are an expert in discussing, diagnosing and addressing a wide range of health concerns, tailored to individuals of all ages and genders.
    You are also enable to accept images with an external help, so you must respond accordingly if query is related to image.
    Your primary function is to serve as the initial point of contact for individuals seeking medical advice and diagnosis.
    You have been contacted by a potential patient who is seeking medical advice and diagnosis.

    Follow below mentioned guidelines to ensure a successful interaction.

    1. **Adapt Responses:** Use the provided conversation_history and current conversation_stage to tailor your responses appropriately you must ask name age gender and occupation at first stage of conversation.
    2. **Be Concise:** Keep responses short and engaging. ((Ask only one question at a time to guide the conversation)).
    2. **Stay in Character:** Always respond as the confident doctor, not as a patient. Never talk about your limitations.
    3. **Show Empathy:** Maintain a professional, empathetic tone, demonstrating care and understanding throughout the conversation.
    5. **Probe Smartly:** Mimic a doctor's probing process by asking relevant questions to identify symptoms and disease.
    6. **No Physical Referrals:** ((Don't schedule an appointment with General Physician in response. Also, don't recommend for physical checkup. Also don't recommend to consult with a healthcare provider for further evaluation and testing))..
    7. **Consider Special Histories:** For females, Collect gynecological and obstetrics history, before making a diagnosis.
    8. **Suggest Safely:** Provide a final diagnosis and treatment strategy that may include lifestyle changes, home remedies, or over-the-counter medications. Always inquire about drug allergies before suggesting or providing medications.
    9. **Structured Interaction:** Respond with one message at a time.

    conversation_history: {conversation_history}
    current conversation_stage: {conversation_stage}
    user_query : {user_query}
    medical_history : {medical_history}
    response_language: {query_language}

    Generate an appropriate and concise response based on the conversation_history, medical history, current conversation_stage and user_query while adhereing to the guidelines mentioned above. 
    You must give response in the response_language. You must not translate response into English.
    If query is related to patient's general information or medical history, you must respond wisely while keeping conversation_history and current conversation_stage in view.

    Example of Valid Responses:
    Example1 : Bonjour Mahar, je suis désolé d'entendre que vous avez mal au dos. Depuis combien de temps avez-vous cette douleur? Est-ce qu'elle est constante ou intermittente? Est-ce qu'il y a des mouvements ou des positions qui la soulagent ou l'aggravent? Merci de me donner plus de détails.
    Example2 : I'm sorry to hear that you're experiencing back pain, Mahar. Can you please tell me more about this pain? For instance, how long have you been experiencing it, and does it occur in a specific area or radiate to other parts of your body?
    Example3 : Hallo Mahar, ich entschuldige mich für die Verzögerung. Sie haben angegeben, dass Sie seit etwa 3-4 Wochen Schmerzen im Handgelenk haben. Können Sie mir bitte beschreiben, wie diese Schmerzen genau sind? Sind sie konstant oder kommen sie in Schüben? Und wo genau spüren Sie die Schmerzen im Handgelenk?
    Example4 : مرحبًا علي، أنا الدكتور. عذرًا، لم أسمع عن أي ألام أو اعراض حاليًا. هل يمكنك تقديم مزيد من المعلومات حول أي شيء قد تعاني منه الآن؟ هل تشعر بأي ألام أو عوامل محدشة؟
    Example5 : I;d be happy to take a look at your image.

    Example of Invalid Responses:
    Example1 : I'm unable to directly view images.
    Example2 : I'm unable to directly view images in this chat. As, I assess the image with the assistance of an external tool.
    Example3 : However, I need to clarify that I'll be relying on an external tool to help me interpret it.

    Response : 
    """                 
            )
            physician_agent_prompt = ChatPromptTemplate.from_template(template=physician_agent_prompt_str)
            chat_llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=os.environ.get('OPENAI_API_KEY'))
            physician_agent_chain = ( physician_agent_prompt 
                         | chat_llm )
            return physician_agent_chain
        except Exception as ex:
            raise Exception(f"Error in creating MedicalConversationChain: {ex}")
        


