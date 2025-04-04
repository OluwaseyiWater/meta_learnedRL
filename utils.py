import pickle

def save_model(path, params, st):
    with open(path, 'wb') as f:
        pickle.dump({'params': params, 'st': st}, f)
    print("Parameters saved to ", path)
    
def load_model(path):
  with open(path, 'rb') as f:
    data = pickle.load(f)
  params = data['params']
  st = data['st']
  print("Parameters loaded from ", path)
  return params, st