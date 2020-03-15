from pomegranate import *
import numpy

age_below_25 = DiscreteDistribution({'True': 0.35, 'False': 0.65})

has_family = ConditionalProbabilityTable(
    [
        ['True','True', 0.3],
        ['True','False', 0.7],
        ['False','True', 0.8],
        ['False','False', 0.2],
    ],
    [age_below_25]
    )

expensive_car = ConditionalProbabilityTable(
    [
        ['True','True', 0.6],
        ['True','False', 0.4],
        ['False','True', 0.4],
        ['False','False', 0.6],
    ],
    [age_below_25]
    )

long_experience = ConditionalProbabilityTable(
    [
        ['True','True', 0.35],
        ['True','False', 0.65],
        ['False','True', 0.66],
        ['False','False', 0.34],
    ],
    [age_below_25]
    )

previous_accident = ConditionalProbabilityTable(
    [
        ['True','True', 0.5],
        ['True','False', 0.5],
        ['False','True', 0.6],
        ['False','False', 0.4],
    ],
    [age_below_25]
    )


car_accident = ConditionalProbabilityTable(
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
[has_family, expensive_car, long_experience, previous_accident]
)

car_accident_fatality = ConditionalProbabilityTable(
    [
        ['True','True', 0.4],
        ['True','False', 0.6],
        ['False','True', 0.05],
        ['False','False', 0.95],
    ],
    [car_accident]
    )

insurance_pays = ConditionalProbabilityTable(
    [
        ['True','True', 0.4],
        ['True','False', 0.6],
        ['False','True', 0.2],
        ['False','False', 0.8],
    ],
    [car_accident]
    )


s1 = State(age_below_25, name="age_below_25")
s2 = State(expensive_car, name="expensive_car")
s3 = State(car_accident, name="car_accident")
s4 = State(has_family, name="has_family")
s5 = State(long_experience, name="long_experience")
s6 = State(previous_accident, name="previous_accident")
s7 = State(car_accident_fatality, name="car_accident_fatality")
s8 = State(insurance_pays, name="insurance_pays")


model = BayesianNetwork("Accident based on various factors")
model.add_states(s1, s2, s3, s4, s5, s6, s7, s8)
model.add_edge(s1, s4)
model.add_edge(s1, s2)
model.add_edge(s1, s5)
model.add_edge(s1, s6)
model.add_edge(s4, s3)
model.add_edge(s2, s3)
model.add_edge(s5, s3)
model.add_edge(s6, s3)
model.add_edge(s3, s7)
model.add_edge(s3, s8)
model.bake()


beliefs = model.predict_proba({ 'car_accident' : 'True'})
beliefs = map(str, beliefs)
print("\n".join( "{} {}".format(state.name, belief) for state, belief in zip(model.states,beliefs)))

print("################################")
beliefs = model.predict_proba({ 'car_accident' : 'False', 'expensive_car': 'True'})
beliefs = map(str, beliefs)
print("\n".join( "{} {}".format(state.name, belief) for state, belief in zip(model.states,beliefs)))


print("################################")
beliefs = model.predict_proba({ 'age_below_25' : 'True', 'expensive_car': 'True'})
beliefs = map(str, beliefs)
print("\n".join( "{} {}".format(state.name, belief) for state, belief in zip(model.states,beliefs)))



print("################################")
print(model.probability(  numpy.array(['True','True','True','True','True','True','True','True'], ndmin=2)  ))