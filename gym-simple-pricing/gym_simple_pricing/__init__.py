import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(id = 'pricing-base-v0', entry_point = 'gym_simple_pricing.envs:PricingBaseEnv')
register(id = 'pricing-model1-v0', entry_point = 'gym_simple_pricing.envs:PricingModel1Env')
register(id = 'pricing-model2-v0', entry_point = 'gym_simple_pricing.envs:PricingModel2Env')


# the id variable here is what we will pass into gym.make() to call our environment
