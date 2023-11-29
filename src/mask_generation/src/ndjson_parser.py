import ndjson
import argparse
import pandas as pd
from tqdm.auto import tqdm


parser = argparse.ArgumentParser(
	prog='.ndjson Parser',
	description='A script to parse .ndjson files from LabelBox'
	)
	
parser.add_argument(
    '-f', 
    '--filename', 
    required=True
    )
    
parser.add_argument(
    '-o', 
    '--output_file', 
    required=True
    )
    
parser.add_argument(
    '-t', 
	'--type', 
	choices=['ImagePoint', 'ImageBoundingBox', 'classification'], 
	help="Choose between ['ImagePoint', 'ImageBoundingBox', 'classification']",
	required=True)
parser.add_argument('-V', '--verbose', help='Verbose output if passed', action='store_true')

args = parser.parse_args()
print(args)

def parse_ndjson(filename=args.filename, a_type=args.type):
	with open(filename) as f:
		data = ndjson.load(f)

	if a_type == 'ImagePoint':
		results = pd.DataFrame(columns=['img_id', 'class', 'x', 'y'])
	elif a_type == 'ImageBoundingBox':
		results = pd.DataFrame(columns=['img_id', 'class', 'top', 'left', 'height', 'width'])
	elif a_type == 'classification':
		pass
	kind = "point" if a_type == 'ImagePoint' else "bounding_box"

	for row in tqdm(data):
		df_annot = pd.DataFrame()
		img_id = row['data_row']['external_id']
		# print(img_id)
		img_h = row['media_attributes']['height']
		img_w = row['media_attributes']['width']

		try:
			projects = row['projects']
			project_id = list(projects.keys())[0]
		except KeyError:
			print(KeyError("'projects' is not in the keys of the row dictionary"))

		annot = projects[project_id]['labels'][0]['annotations']
		if (a_type == 'ImagePoint') | (a_type == 'ImageBoundingBox'):
			objects = annot['objects']
			for obj in objects:
				if obj['annotation_kind'] == args.type:
					# print(args.type)
					
					sample_dict = obj[kind]
					sample_dict['class'] = obj['name']
					sample_dict['img_id'] = img_id

					sample_df = pd.DataFrame(sample_dict, index=[0])
					df_annot = pd.concat([df_annot, sample_df], ignore_index=True)
		results = pd.concat([results, df_annot], ignore_index=True)

	results.to_csv(args.output_file)
	return results


if __name__ == '__main__':
	parse_ndjson()
