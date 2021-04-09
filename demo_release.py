
import argparse
import base64
import io

import matplotlib.pyplot as plt
from numpy import asarray

from colorizersDl import *
from PIL import Image
from flask import Flask, jsonify, Response
from flask import request
from io import BytesIO

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
@app.route('/ArtisticStyle', methods=['POST', 'GET'])
def hello_world():
	content_url = request.form.get('realImg')
	input_image = Image.open(BytesIO(base64.b64decode(content_url)))

	parser = argparse.ArgumentParser()
	parser.add_argument('-i','--img_path', type=str, default=input_image)
	parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
	parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
	opt = parser.parse_args()
	# load colorizersDl
	colorizer_siggraph17 = siggraph17(pretrained=True).eval()
	if(opt.use_gpu):
		colorizer_siggraph17.cuda()

	# default size to process images is 256x256
	# grab L channel in both original ("orig") and resized ("rs") resolutions
	#img = load_img(opt.img_path)
	data = asarray(input_image)

	(tens_l_orig, tens_l_rs) = preprocess_img(data, HW=(256,256))
	if(opt.use_gpu):
		tens_l_rs = tens_l_rs.cuda()

	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
	out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

	plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)



	# plt.subplot(2,2,2)
	# plt.imshow(img_bw)
	# plt.title('Input')
	# plt.axis('off')
	#
	#
	#
	# plt.subplot(2,2,4)
	# plt.imshow(out_img_siggraph17)
	# plt.title('Output (SIGGRAPH 17)')
	# plt.axis('off')
	# plt.show()
	# out_img_siggraph17 = np.clip(out_img_siggraph17, 0, 255).astype("uint8")
	output_image = Image.fromarray(out_img_siggraph17, "RGB")
	buffered = BytesIO()
	output_image.save(buffered, format="PNG")
    #base64
	img_str = base64.b64encode(buffered.getvalue()).decode()


	# buf = io.BytesIO(msg)
	# img = Image.open(buf)
	# plt.imshow(img)
	# plt.show()
	print(img_str)
	return jsonify(img_str)

if __name__ == "__main__":
	app.run(host='192.168.1.3',port=5000)