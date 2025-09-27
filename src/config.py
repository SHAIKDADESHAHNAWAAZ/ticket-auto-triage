# Configuration settings
MODEL_PATH = 'models/ticket_classifier.pkl'
DATA_PATH = 'data/support_tickets.csv'

# Model parameters
TFIDF_MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# API settings
API_HOST = '0.0.0.0'
API_PORT = 5000
DEBUG_MODE = True
