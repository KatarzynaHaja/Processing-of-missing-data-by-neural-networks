# import pandas as pd
# headers=['Method', 'Num_sample', "Epoch_number", "Gamma", 'Result']
# data = pd.read_csv("results.csv", names=headers)
# for header in headers:
#     if header != 'Method':
#         data[header] = data[header].str.split(':').str.get(-1)
#
# last_layer_results = data.where(data['Method'] == 'last_layer').dropna().sort_values(by='Result')
# first_layer_results =  data.where(data['Method'] == 'first_layer').dropna().sort_values(by='Result')
# different_cost_results =  data.where(data['Method'] == 'different_cost').dropna().sort_values(by='Result')
#
# last_layer_results.to_csv('last_layer_results', index=False)
# first_layer_results.to_csv('first_layer_results', index=False)
# different_cost_results.to_csv('different_cost_results', index=False)
# data.sort_values(by='Result').to_csv('sorted_results_ac_fc', index=False)

