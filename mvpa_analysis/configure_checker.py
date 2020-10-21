# File: configure_checker.py
# Aim: Check the configuration

from tools import Configure

config = Configure()
settings = config.getall()
print(settings)
