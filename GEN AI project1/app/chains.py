import  os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv


load_dotenv()

#
# if __name__ == "__main__":
#     print(os.getenv("groq_api_key"))

class Chain:
    def __init__(self):
        self.llm =  ChatGroq( temperature=0,groq_api_key= os.getenv("groq_api_key"),model_name= "llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ###SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ###INSTRUTIONS:
           The Scrapped text is from a career's page of a website
           Your job is to extract the job posting and return them into JSON format
           containing the following keys:
           'role', 'skills', 'desccription'.
           Only return in Valid JSON.
           ###Valid JSON(NO PREAMBLE):
           """
        )

        chain_extract = prompt_extract | self.llm

        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big, Unable to parse job. ")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ###SCRAPED TEXT FROM WEBSITE:
            {job_description}
            ### INSTRUCTION:
      You are Santayo, a business development executive at Amazon. TCS is an Indian multinational IT services and consulting company.
      It's a part of the Tata Group, one of India's largest and oldest conglomerates.
      Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
      process optimization, cost reduction, and heightened overall efficiency. 
      Your job is to write a cold email to the client regarding the job mentioned above describing the capability of Amazon
      in fulfilling their needs.
      Also add the most relevant ones from the following links to showcase TCS's portfolio: {link_list}
      Remember you are Santayo, BDE at TCS. 
      Do not provide a preamble.
      ### EMAIL (NO PREAMBLE):
           """
        )

        chain_email = prompt_email | self.llm

        res = chain_email.invoke(input={"job_description": str(job), "link_list": links})
        return res.content


if __name__ == "__main__":
    print(os.getenv("groq_api_key"))