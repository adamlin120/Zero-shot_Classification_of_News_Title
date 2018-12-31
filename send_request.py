# encoding: utf-8
from pprint import pprint
import requests
import argparse


def predict_result(payload_json, api_url):
    # Submit the request.
    r = requests.post(api_url, json=payload)
    r = r.json()

    # Ensure the request was successful.
    if r['req_id'] == payload_json['data']['req_id']:
        return r
    # Otherwise, the request failed.
    else:
        print('Request failed: Maybe ID not match')
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('--title', default='')
    parser.add_argument('--tags', nargs='+', default='')
    parser.add_argument('--api_url', default='http://140.112.252.117:5010/predict')
    args = parser.parse_args()

    # using user defined request
    if args.title != '' and args.tags != '':
        payload = {'data': {'title': args.title, 'tags': list(map(lambda x: x.strip(), args.tags)), 'req_id': 'IamID'}}
        payload['data']['num_tags'] = len(payload['data']['tags'])
        print("\nUsing User defined payload: {}\n".format(payload))
    # using testing request
    else:
        payload = {'data': {'title': '測試測試測試測試測試測試測試測試', 'tags': ['問', '號號號', 'Hi'], 'req_id': 'abcedd', 'num_tags': 3}}
        print("\nUsing default testing payload: {}\n".format(payload))

    r = predict_result(payload, args.api_url)
    pprint(r)
