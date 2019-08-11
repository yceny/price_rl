import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(id = 'pricing-simple-v1', entry_point = 'gym_simple_pricing.envs:PricingSimpleEnv')
register(id = 'pricing-base-v1', entry_point = 'gym_simple_pricing.envs:PricingBaseEnv')

register(id = 'pricing-ext-v1', entry_point = 'gym_simple_pricing.envs:PricingExt1Env')
register(id = 'pricing-ext-v2', entry_point = 'gym_simple_pricing.envs:PricingExt2Env')
register(id = 'pricing-ext-v3', entry_point = 'gym_simple_pricing.envs:PricingExt3Env')


# the id variable here is what we will pass into gym.make() to call our environment
