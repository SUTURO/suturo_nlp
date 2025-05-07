import json
from word2number import w2n
import re
from typing import List
from pydantic import BaseModel
from ollama import chat # type: ignore

entities = [
    "apple juice", "beer", "bottle of wine", "cafe au lait", "coffee", "coffee can", "can of coffee", 
    "coffee with milk", "coke", "can of coke", "cola", "cola can", "cup of coffee", "cup of coffee with milk", 
    "milk coffee", "cup of tea", "gin tonic", "ginger ale", "glass of milk", "glass of mineral water", 
    "glass of water", "ice tea bottle", "bottle of ice tea", "iced coffee", "iced tea", "jug of milk", 
    "juice box", "juice pack", "pack of juice", "lemonade", "milk", "milk pack", "mineral water", 
    "orange juice", "orange juice box", "box of orange juice", "sprite", "tea", "tropical juice bottle", 
    "bottle of tropical juice", "water", "wine bottle", "wine glass", "glass of wine", "milk bottle", 
    "bottle of milk", "ice tea", "ice tea can", "can of ice tea", "water bottle", "bottle of water", 
    "big coke", "big cola", "fanta", "fanta can", "can of fanta", "dubbelfris", "red bull", 
    "lactosefree milk", "mezzo mix", "oat milk"
]

def switch(case, response, context):
    '''
    Manual Implementation of switch(match)-case because python3.10 first implemented one, this uses 3.8.
    
    Args:
        case: The intent parsed from the response
        response: The formatted .json from the record function

    Returns:
        The function corresponding to the intent
    '''
    return {
        "Receptionist": lambda: Receptionist.receptionist(response,context),
        "Order": lambda: Restaurant.order(response,context),
        "Hobbies": lambda: Receptionist.hobbies(response,context),
        "affirm": lambda: context["pub"].publish(f"<CONFIRM>, True"),
        "deny": lambda: context["pub"].publish(f"<DENY>, False")
    }.get(case, lambda: context["pub"].publish(f"<NONE>"))()

def getData(response):
        """
        Function for extracting names, drinks, and foods from entities in a JSON response.

        Args:
            response: The JSON string containing entities.
        
        Returns:
            A JSON string categorizing names, drinks, and foods.
        """
        
        # Parse the response JSON string into a dictionary
        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")

        drinks = []
        foods = []
        names = []
        interests = []

        cachedNumer = 0
        falseNumber = False

        entities = response_dict.get("entities", [])
        if not isinstance(entities, list):
            raise ValueError("Expected 'entities' to be a list in the response.")

        # Filtering the entities list for drink, food and NaturalPerson and the amount of each
        for ent in entities:
            if isinstance(ent, dict):
                entity = ent.get("entity")
                value = ent.get("value")
                value = value.strip()

                #print(value)
                number = ent.get("numberAttribute")
 
                if entity == "drink":
                    if not number:
                        drinks.append((value, 1))
                    else:
                        drinks.append((value, number[0] if type(number[0]) == int else w2n.word_to_num(number[0])))
                elif entity == "food":
                    if not number:
                        foods.append((value,1))
                    else:
                        foods.append((value, number[0] if type(number[0]) == int else w2n.word_to_num(number[0])))
                elif entity == "NaturalPerson":
                    names.append(value)
                elif entity == "Interest":
                    interests.append(value)
                
        # Build the .json
        values= {"names": names, "drinks": drinks, "foods": foods, "hobbies": interests}
        return json.dumps(values) 

def replace(false_term):
    # Define the schema for the response
    class Entity(BaseModel):
        name: str

    class ReplacementList(BaseModel):
        replacements: List[Entity]

    response = chat(
        model='gemma3',
        messages=[{
            'role': 'user',
            'content': "You will be given a term and a list of entities. Please select an entity that sounds closest to the given term when pronounced and return it in json format. For example: the term 'wet boy' should be replaced with 'red bull'. Entities:" + ", ".join(entities) + ". Term: " + false_term
        }],
        format=ReplacementList.model_json_schema(),
        options={'temperature': 1},
    )

    # Validate response
    response = ReplacementList.model_validate_json(response.message.content)
    replacement_term = response.replacements[0].name 
    return replacement_term

class Receptionist:
    """
    A class for the Receptionist challenge. Extracts the entities and their values and checks
    whether they are a NaturalPerson, a Drink or Food. In the end a string is published
    containing the name and the favorite drink of a person.

    Methods:
        receptionist(response, context):
            Processes the data given by getData and publishes a string
            with the extracted NaturalPerson and drink.
        hobbies(response, context):
            Processes the data given by getData and publishes a string
            with the extracted Interest.
    """
    
    def receptionist(response,context):
        '''
        Function for the receptionist task. Extracts the first name and drink in the list of names 
        and drinks the getData() function may return and publishes those. 

        Args:
            response: Formatted .json from record function.
            context:
                pub: a ROS publisher object to publish processed results to a specified topic.
        '''
        data = json.loads(getData(response)) 

        name = data.get("names")
        if name:
            name = name[0]
            
        drink = data.get("drinks")
        drink = [x[0] for x in drink][0] if drink else None

        print('Original drink:', drink) 

        if drink not in entities and drink is not None:
            drink = replace(drink)

        print('Replaced drink:', drink)

        context["pub"].publish(f"<GUEST>; {name}; {drink}")

    def hobbies(response, context):
        '''
        Function to extract interests from the list of hobbies the getData() function may return
        and publishes those.

        Args:
            response: Formatted .json from record function.
            context:
                pub: a ROS publisher object to publish processed results to a specified topic.
        '''
        data = json.loads(getData(response))

        hobby = data.get("hobbies")

        context["pub"].publish(f"<INTERESTS>; {hobby}")


class Restaurant:
    """
    A class for the Restaurant challenge. Extracts the entities and their values and checks
    whether they are a NaturalPerson, a Drink or Food. In the end a string is published
    containing the food and drink a person woukd like to order.

    Methods:
        restaurant(response, context):
            Processes the date given by getData and publishes a string with the extracted NaturalPerson and Drink.
    """

    def order(response, context):
        """
        Function for the order task.

        Args:
            response: JSON string with entities from `nluInternal`.
            context: Context dictionary, including:
                pub: ROS publisher object to publish results to a specified topic.
        """
        data = json.loads(getData(response))

        food = data.get("foods")

        drink = data.get("drinks")

        # Publish the order
        context["pub"].publish(f"<ORDER>, {food}, {drink}")


