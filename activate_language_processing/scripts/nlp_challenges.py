import json
from word2number import w2n
import re

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

def replace_word_and_next(text, target_word, replacement):
    # Regular expression to find the target word followed by another word
    pattern = rf"\b{target_word}\s+\w+\b"
    return re.sub(pattern, replacement, text)

"""
def is_number(value):
    
    Check if a given string is a number (either numeral or word).

    Args:
        value: a string, that might be a digit or word representation of a number.
    
    Returns:
        True if the string is a textual representation of a number (or a digit) else False.
    
    try:
     
        w2n.word_to_num(value)  
        return True
    except ValueError:
        return value.isdigit() 
"""

"""        
def to_number(value):
    
    Converts a number in words or numerals to an integer.

    Args:
        value: A string that contains a digit or word representation of a number.
        
    Returns:
        The input number as an integer.
    
    return int(value) if value.isdigit() else w2n.word_to_num(value)
"""

blacklist = {"states": "steaks", "slates": "steaks", "slaves": "steaks", "stakes": "steaks", 
        "red boy": "red bull", "redbull": "red bull", "whetball": "red bull", "whet ball": "red bull",
        "red bullseye": "red bull", "red balloon": "red bull", "red bullet": "red bull", "bed pull": "red bull",
        "let bull": "red bull", "wet bull": "red bull","dead bull": "red bull","red boot": "red bull","red bell": "red bull",
        "red pool": "red bull","red bowl": "red bull","read bull": "red bull","red pull": "red bull","red ball": "red bull",
        "rad bull": "red bull","rat bull": "red bull","red full": "red bull","red wool": "red bull","rip bull": "red bull",
        "wetball": "red bull", "wet ball": "red bull", "wet": "red bull", "boy": "red bull", "wet bull": "red bull"}

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

                if value == "boy":
                    entity = "drink"

                if value in blacklist:
                    value = blacklist.get(value)

                #print(value)
                number = ent.get("numberAttribute")
                
                """
                if is_number(value):
                    cachedNumer = to_number(value)
                    falseNumber = True
                    continue
                """
                    
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


