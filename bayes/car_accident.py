from pomegranate import *
import numpy

class CarAccidentProbabilityCalculator:
    def __init__(self):

        self.age_below_25 = DiscreteDistribution({'True': 0.25, 'False': 0.75})

        self.has_family = ConditionalProbabilityTable(
            [
                ['True','True', 0.3],
                ['True','False', 0.7],
                ['False','True', 0.6],
                ['False','False', 0.4],
            ],
            [self.age_below_25]
            )

        self.expensive_car = ConditionalProbabilityTable(
            [
                ['True','True', 0.6],
                ['True','False', 0.4],
                ['False','True', 0.4],
                ['False','False', 0.6],
            ],
            [self.age_below_25]
            )

        self.long_experience = ConditionalProbabilityTable(
            [
                ['True','True', 0.35],
                ['True','False', 0.65],
                ['False','True', 0.66],
                ['False','False', 0.34],
            ],
            [self.age_below_25]
            )

        self.previous_accident = ConditionalProbabilityTable(
            [
                ['True','True', 0.5],
                ['True','False', 0.5],
                ['False','True', 0.6],
                ['False','False', 0.4],
            ],
            [self.age_below_25]
            )

        self.car_accident = ConditionalProbabilityTable(
        [       
                # Fam    Car    Exp    Acc      
                ['True','True','True','True','True', 0.4],
                ['True','True','True','True','False', 0.6],

                ['True','True','True','False','True', 0.3],
                ['True','True','True','False','False', 0.7],

                ['True','True','False','True','True', 0.6],
                ['True','True','False','True','False', 0.4],

                ['True','True','False','False','True', 0.5],
                ['True','True','False','False','False', 0.5],

                ['True','False','True','True','True', 0.4],
                ['True','False','True','True','False', 0.6],

                ['True','False','True','False','True', 0.2],
                ['True','False','True','False','False', 0.8],

                ['True','False','False','True','True', 0.4],
                ['True','False','False','True','False', 0.6],

                ['True','False','False','False','True', 0.3],
                ['True','False','False','False','False', 0.7],

                ['False','True','True','True','True', 0.66],
                ['False','True','True','True','False', 0.34],

                ['False','True','True','False','True', 0.5],
                ['False','True','True','False','False', 0.5],

                ['False','True','False','True','True', 0.8],
                ['False','True','False','True','False', 0.2],

                ['False','True','False','False','True', 0.7],
                ['False','True','False','False','False', 0.3],

                ['False','False','True','True','True', 0.6],
                ['False','False','True','True','False', 0.4],

                ['False','False','True','False','True', 0.3],
                ['False','False','True','False','False', 0.7],

                ['False','False','False','True','True', 0.67],
                ['False','False','False','True','False', 0.33],

                ['False','False','False','False','True', 0.5],
                ['False','False','False','False','False', 0.5]

        ],
        [   self.has_family, 
            self.expensive_car, 
            self.long_experience, 
            self.previous_accident
        ]
        )

        self.car_accident_fatality = ConditionalProbabilityTable(
            [
                ['True','True', 0.2],
                ['True','False', 0.8],
                ['False','True', 0.05],
                ['False','False', 0.95],
            ],
            [self.car_accident]
            )

        self.insurance_pays = ConditionalProbabilityTable(
            [
                ['True','True', 0.4],
                ['True','False', 0.6],
                ['False','True', 0.1],
                ['False','False', 0.9],
            ],
            [self.car_accident]
            )


        self.s1 = State(self.age_below_25, name="age_below_25")
        self.s2 = State(self.expensive_car, name="expensive_car")
        self.s3 = State(self.car_accident, name="car_accident")
        self.s4 = State(self.has_family, name="has_family")
        self.s5 = State(self.long_experience, name="long_experience")
        self.s6 = State(self.previous_accident, name="previous_accident")
        self.s7 = State(self.car_accident_fatality, name="car_accident_fatality")
        self.s8 = State(self.insurance_pays, name="insurance_pays")


        self.model = BayesianNetwork("Accident based on various factors")
        self.model.add_states(self.s1, self.s2, self.s3, 
                    self.s4, self.s5, self.s6, self.s7, self.s8)
        self.model.add_edge(self.s1, self.s4)
        self.model.add_edge(self.s1, self.s2)
        self.model.add_edge(self.s1, self.s5)
        self.model.add_edge(self.s1, self.s6)
        self.model.add_edge(self.s4, self.s3)
        self.model.add_edge(self.s2, self.s3)
        self.model.add_edge(self.s5, self.s3)
        self.model.add_edge(self.s6, self.s3)
        self.model.add_edge(self.s3, self.s7)
        self.model.add_edge(self.s3, self.s8)
        self.model.bake()

    def predict_probabilities_based_on_facts(self, facts):
        """
        facts: dict, a dictionary of known facts
        """
        
        beliefs = self.model.predict_proba(facts)
        beliefs = map(str, beliefs)
        ret = "\n".join( "{} {}".format(state.name, belief) \
                for state, belief in zip(self.model.states, beliefs))
        print(ret)
        return ret

    def probability_of_all_events_happening(self, event_array):
        """
        event_array: array, an array of the values of the nodes
        """
        if len(event_array) != 8:
            raise "Event array given is too short"
        ret = self.model.probability(numpy.array(event_array, ndmin=2))
        print(ret)
        return ret



ca = CarAccidentProbabilityCalculator()

ca.predict_probabilities_based_on_facts(
    {'car_accident':'True'})
print("______________________________________________")
ca.predict_probabilities_based_on_facts(
    {'car_accident':'False',
    'expensive_car':'True'})
print("______________________________________________")
ca.predict_probabilities_based_on_facts(
    {'age_below_25':'True',
    'has_family':'True'}
    )
print("______________________________________________")
ca.probability_of_all_events_happening(
    ['True','True','True','True',
    'True','True','True','False'])

print("______________________________________________")
ca.probability_of_all_events_happening(
    ['True','True','True','True',
    'True','True','True','True'])

