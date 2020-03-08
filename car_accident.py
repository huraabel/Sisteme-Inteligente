from pomegranate import *

age_below_25 = DiscreteDistribution({'True': 0.65, 'False': 0.35})
expensive_car = DiscreteDistribution({'True': 0.55, 'False': 0.45})
accident = ConditionalProbabilityTable(
[
        ['True','True','True', 0.8],
        ['True','True','False', 0.2],
        ['True','False','True', 0.6],
        ['True','False','False', 0.4],

        ['False','True','True', 0.5],
        ['False','True','False', 0.5],
        ['False','False','True', 0.4],
        ['False','False','False', 0.6]

],
[age_below_25, expensive_car]
)
       

s1 = State(age_below_25, name="age_below_25")
s2 = State(expensive_car, name="expensive_car")
s3 = State(accident, name="accident")

model = BayesianNetwork("Accident based on age and car driven")
model.add_states(s1, s2, s3)
model.add_edge(s1, s3)
model.add_edge(s2, s3)
model.bake()

beliefs = model.predict_proba({ 'accident' : 'True'})
beliefs = map(str, beliefs)
print("\n".join( "{} {}".format(state.name, belief) for state, belief in zip(model.states,beliefs)))


beliefs = model.predict_proba({ 'accident' : 'False', 'expensive_car': 'True'})
beliefs = map(str, beliefs)
print("\n".join( "{} {}".format(state.name, belief) for state, belief in zip(model.states,beliefs)))