# Importing necessary libraries
from langchain_google_genai import ChatGoogleGenerativeAI # JSON format is not working with gemini model so i used llama
from langchain_groq import ChatGroq # Free Access
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
"""
    TypedDict: A class that allows you to define a dictionary with specific keys and types for those keys.
    Annotated: A class that allows you to add metadata to a type.
    Optional: A class that allows you to define a type that can be not maindatory or none.
    Literal: A class that allows you to define a type that can only be one of a set of values. Its fixed value.
"""

# Loading env variables
load_dotenv()

# Creating Schema
class Review(TypedDict):
    theams: Annotated[list[str], "The list of all the key theams of the review"]
    summary: Annotated[str, "A short summary of the review"]
    sentiment: Annotated[Literal["Positive", "Negative"], "The sentiment of the review"]
    pros: Annotated[list[str], "The list of pros of the review"]
    cons: Annotated[list[str], "The list of cons of the review"]
    name: Annotated[Optional[str], "The name of the reviewer"]

# Model Building
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=None)

# Creating Structured Output object
structured_model = model.with_structured_output(Review)

# Passing Prompt to model
prompt = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, its an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether Im gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsungs One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Weight and size are a bit uncomfortable for one-handed use
Samsungs One UI still comes with bloatware
Price tag is steep
                                 
Review by Dexter Morgan
"""
result = structured_model.invoke(prompt)

# Printing the result
print(result)