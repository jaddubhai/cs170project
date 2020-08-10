import os

input_dir = "inputs"
output_dir = "out"
remaining = "./remaining_files.txt"
for inp in os.listdir(input_dir):
	if not os.path.exists(output_dir + "/" + inp.replace('in', 'out')):
		with open(remaining, 'a') as f:
			f.writelines(inp)
			f.writelines("\n")