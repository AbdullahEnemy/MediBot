import os  # Standard library module for interacting with the operating system
import logging  # Standard library module for logging
from dotenv import load_dotenv 
# For loading environment variables from a .env file
from fastapi import FastAPI, status, Body
# FastAPI components
from fastapi.middleware.cors import CORSMiddleware  
# Middleware for handling CORS
import uvicorn  # ASGI server for running FastAPI applications
from utils import *  # Custom module 
from fastapi.responses import JSONResponse # Response classes for FastAPI
import json  # Standard library module for JSON manipulation
import requests
import time


# model name for env
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

# Load environment variables from a .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s:%(funcName)s')
file_handler = logging.FileHandler('gppod_medicalGPT.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Initialize FastAPI application
app = FastAPI()

# Set up CORS middleware to allow requests from specified origins
origins = ["*"]  
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Root endpoint
@app.get("/", response_class=JSONResponse)
async def index() -> JSONResponse:
    try:
        # Log info for entering index endpoint
        logger.info("Entering index endpoint")
        message = "Welcome To MediBOT"
        # Return JSON response
        return JSONResponse(content={"succeeded": True, "message": message, "httpStatusCode": status.HTTP_200_OK})
    except Exception as e:
        # Log error if exception occurs
        logger.error(f"Error in index endpoint: {e}")
        # Return error response
        return JSONResponse(content={"succeeded": False, "message": "Failed to start the conversation", "httpStatusCode": status.HTTP_500_INTERNAL_SERVER_ERROR}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)






@app.post("/qnaConversation", response_class=JSONResponse)
async def qna_conversation( query: ConversationQuery = Body(...)):
    try:
        history_list = []
        # Log info for entering qna_conversation endpoint
        logger.info("Entering qna_conversation endpoint")   
        # Extract data from query
        chat_id = query.chat_id
        user_query = query.user_query
        Id = query.Id
      # Log user ID 
        logger.info(f"Chat ID: {chat_id}")
        physician_name = "William" if Id == 0 else "Elizabeth"
        patientName = query.name
        patientAge = query.age
        patientGender = query.gender
        patientOccupation = query.occupation
        firstMessage = query.firstMessage
        patientId = query.patientId
        language_code = query.language_code

        history_list = ""
        logger.info(f"Successfully got medical history for patient id: {patientId}")

        history_list.append("Hello, I am Dr. , a General Physician. How can I assist you today? Could you please tell me your name, age, gender, and occupation?")
        history_list.append(f"My name is {patientName} and my age is {patientAge}, and my gender is {patientGender} and my occupation is {patientOccupation}")
        

        # query_lang = detect_language(user_query)
        stage_analyzer_chain = ConversationStageAnalyzer.from_openai_llm(llm_name=OPENAI_MODEL_NAME)
        start_time = time.time()
        stage_and_history_dict = await create_conv_stage_and_history_pair(history_list, stage_analyzer_chain=stage_analyzer_chain, chat_id=chat_id)
        end_time = time.time()
        elapsed_time = end_time - start_time
        conv_stage = int(list(stage_and_history_dict.keys())[0])
        logger.info(f"Succesfully got conversation stage: {conv_stage}")
        conversation_chain = MedicalConversationChain.from_openai_llm(llm_name=OPENAI_MODEL_NAME)
        physician_agent_chain = await conversation_chain.ainvoke({'physician_name': physician_name, 'conversation_stage': conv_stages_summary_dict[conv_stage], 'conversation_history': stage_and_history_dict[str(conv_stage)], 'user_query': user_query, 'medical_history': history_list})
        physician_agent_chain = physician_agent_chain.content
        logger.info("Successfully completed the conversation")
        # Return success response with conversation data
        return JSONResponse(content={"succeeded": True, "message": "Successfully completed the conversation", "httpStatusCode": status.HTTP_200_OK, "data": physician_agent_chain}, status_code=status.HTTP_200_OK)
    except Exception as e:
        # Log critical error if exception occurs during conversation
        logger.critical(f"Failed: {e}")
        # Return error response for failed conversation
        return JSONResponse(content={"succeeded": False, "message": "Failed to complete the conversation", "httpStatusCode": status.HTTP_500_INTERNAL_SERVER_ERROR}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        # Log info for exiting qna_conversation endpoint
        logger.info("Exiting qna_conversation endpoint")

    


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9595)