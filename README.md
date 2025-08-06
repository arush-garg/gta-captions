# GTA Caption Generator

Using the `vipulmaheshwari/GTA-Image-Captioning-Dataset` from HuggingFace, I fine-tuned Microsoft's Generative Image2Text Transformer to generate captions for GTA scenes (at this rate, AI might generate GTA 6 before Rockstar Games does).

The model trained is relatively small, so it works on CPU, CUDA and MPS (Apple Metal). However, you might notice slower inference if running on CPU.

To test it out, click [here]()! 


## Set up
1. `pip install -r requirements.txt`
2. Create a `.env` file and add the download path to the model weights. I hosted them on pCloud (it is free and allows download using API, but you need to sign up)
3. Run the app using `streamlit run app.py`
<br><br>
<i>In case you're wondering, I used SSH into Google Colab to train the model using GPU. Check out this [tutorial](https://www.youtube.com/watch?v=wvDFNQNgqS8) for how to do it (though you shouldn't do it on a Free account)</i>