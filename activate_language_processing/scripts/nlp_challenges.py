import json

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

        
        entities = response_dict.get("entities", [])
        if not isinstance(entities, list):
            raise ValueError("Expected 'entities' to be a list in the response.")

        # Filtering the entities list for drink, food and NaturalPerson and the amount of each
        for ent in entities:
            if isinstance(ent, dict):
                entity = ent.get("entity")
                value = ent.get("value")
                number = ent.get("numberAttribute")

                if entity == "drink":
                    if not number:
                        drinks.append((value,1))
                    else:
                        drinks.append((value,number[0]))
                elif entity == "food":
                    if not number:
                        foods.append((value,1))
                    else:
                        foods.append((value,number[0]))
                elif entity == "NaturalPerson":
                    if not number:
                        names.append((value,1))
                    else:
                        names.append((value,number[0]))

        # Build the .json
        values= {"names": names, "drinks": drinks, "foods": foods}
        return json.dumps(values)


class Receptionist:
    """
    A class for the Receptionist challenge. Extracts the entities and their values and checks
    whether they are a NaturalPerson, a Drink or Food. In the end a string is published
    containing the name and the favorite drink of a person.

    Methods:
        receptionist(response, context):
            Processes the date given by getData and publishes a string
            with the extracted NaturalPerson and Drink.
    """
    
    def receptionist(response,context):
        '''
        Function for the receptionist task. 

        Args:
            response: Formatted .json from record function.
            context:
                pub: a ROS publisher object to publish processed results to a specified topic.
        '''
        data = json.loads(getData(response)) 

        name = data.get("names")
        name = [x[0] for x in name][0] if name else None

        drink = data.get("drinks")
        drink = [x[0] for x in drink][0] if drink else None

        context["pub"].publish(f"<GUEST>, {name}, {drink}")


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


