import json


class Receptionist:
    """
    A class for the Receptionist challange. Extracts the entities and their values and checks
    whether they are a NaturalPerson, a Drink or Food. In the end a string is published
    containing the name and the favorite drink of a person.

    Methods:
        getData(response):
            Parses a JSON string to extract entities and filters for 
            NaturalPerson, Drink and Food.

        receptionist(response, context):
            Processes the date given by getData and publishes a string
            with the extracted NaturalPerson and Drink.
    """
    def getData(response):
            """
            Function for extracting names, drinks, and foods from entities in a JSON response.

            Args:
                response: The JSON string containing entities.
            """
            
            # Parse the response JSON string into a dictionary
            try:
                response_dict = json.loads(response)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response: {e}")

            drinks = []
            foods = []
            names = []
            
            # Iterate through the response_dict to process entities
            for role, ent in response_dict.items():
                
                # Filtering the entities list for Drink, Food and NaturalPerson
                if isinstance(ent, dict):
                    entity = ent.get("entity")
                    value = ent.get("value")
                    
                    if entity == "Drink":
                        drinks.append(value)
                    elif entity == "Food":
                        foods.append(value)
                    elif entity == "NaturalPerson":
                        names.append(value)
                    else:
                        pass

            # Build the .json
            list = {"names": names,"drinks": drinks,"foods": foods}
            return json.dumps(list)


    def receptionist(response,context):
        '''
        Function for the receptionist task. 

        Args:
            response: Formatted .json from record function.
            context:
                pub: a ROS publisher object to publish processed results to a specified topic.
        '''
        data = json.loads(Receptionist.getData(response)) 

        name = data.get("names")
        name = name[0] if name else None

        drink = data.get("drinks")
        drink = drink[0] if drink else None

        context["pub"].publish(f"<GUEST>, {name}, {drink}")

   
