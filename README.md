# Zero-shot Classification of News Title



## Getting Started


### Prerequisites

GPU enabled environment is required.

```
numpy
PyTorch >= 1.0
pytorch_pretrained_bert

tqdm
redis
flask
```

## Running the tests

User can get the prediction result by sending request to server.

### Send Request

```
python3 send_request [--title TITLE] [--tags TAGS [TAGS ...]] [--api_url API_URL]
```


## Deployment With

* [Flask](http://flask.pocoo.org/) - The python web framework used
* [Redis](https://redis.io/) - In-memory cache

## Authors

* **YenTing Lin** - [GitHub](https://github.com/adamlin120)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* pytorch_pretrained_bert is cloned from https://github.com/huggingface/pytorch-pretrained-BERT
* Zero-shot classification for text is inspired by https://arxiv.org/abs/1712.05972
