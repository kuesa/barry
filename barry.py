import discord
import asyncio
import random
import re
import argparse
from six import text_type
import json
import os
import train as tr
import sample as sp
import importlib
import sys
import tensorflow as tf

# load SECRET DATA from JSON file
with open('client_info.json') as f:
    client_info = json.load(f)
client_secret = client_info['secret']
client_channel = client_info['channel']

# taken from sample.py and train.py to pass arguments to train files
sampleParser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
sampleParser.add_argument('--save_dir', type=str, default='save',
                          help='model directory to store checkpointed models')
sampleParser.add_argument('-n', type=int, default=500,
                          help='number of characters to sample')
sampleParser.add_argument('--sample', type=int, default=1,
                          help='0 to use max at each timestep, 1 to sample at '
                          'each timestep, 2 to sample on spaces')
sampleParser.add_argument('--prime', type=text_type, default='',
                          help='prime text')
sampleArgs = sampleParser.parse_args()

trainParser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data and model checkpoints directories
trainParser.add_argument('--data_dir', type=str, default='data',
                         help='data directory containing input.txt with training examples')
trainParser.add_argument('--save_dir', type=str, default='save',
                         help='directory to store checkpointed models')
trainParser.add_argument('--log_dir', type=str, default='logs',
                         help='directory to store tensorboard logs')
trainParser.add_argument('--save_every', type=int, default=1000,
                         help='Save frequency. Number of passes between checkpoints of the model.')
# Model params
trainParser.add_argument('--model', type=str, default='lstm',
                         help='lstm, rnn, gru, or nas')
trainParser.add_argument('--rnn_size', type=int, default=128,
                         help='size of RNN hidden state')
trainParser.add_argument('--num_layers', type=int, default=2,
                         help='number of layers in the RNN')
# Optimization
trainParser.add_argument('--seq_length', type=int, default=50,
                         help='RNN sequence length. Number of timesteps to unroll for.')
trainParser.add_argument('--batch_size', type=int, default=50,
                         help="""minibatch size. Number of sequences propagated through the network in parallel.
                            Pick batch-sizes to fully leverage the GPU (e.g. until the memory is filled up)
                            commonly in the range 10-500.""")
trainParser.add_argument('--num_epochs', type=int, default=50,
                         help='number of epochs. Number of full passes through the training examples.')
trainParser.add_argument('--grad_clip', type=float, default=5.,
                         help='clip gradients at this value')
trainParser.add_argument('--learning_rate', type=float, default=0.002,
                         help='learning rate')
trainParser.add_argument('--decay_rate', type=float, default=0.97,
                         help='decay rate for rmsprop')
trainParser.add_argument('--output_keep_prob', type=float, default=1.0,
                         help='probability of keeping weights in the hidden layer')
trainParser.add_argument('--input_keep_prob', type=float, default=1.0,
                         help='probability of keeping weights in the input layer')

if os.path.isfile('save/config.pk1'):
    trainParser.add_argument('--init_from', type=str, default='save', help="")
else:
    trainParser.add_argument('--init_from', type=str, default=None, help="")

trainArgs = trainParser.parse_args()

client = discord.Client()
training = False


@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')


@client.event
async def on_message(message):
    global training
    if client.user.mentioned_in(message):
        sp.sample(sampleArgs, message.content)
        tf.reset_default_graph()
        with open('output/output.txt', 'r') as the_file:
            lines = the_file.read().split('\\r\\n')
            await client.send_message(discord.Object(id=client_channel), lines[1].encode('utf-8').decode('unicode-escape'), tts=bool(random.getrandbits(1)))
    elif message.content.startswith('!record'):
        print('Recording...')
        with open('data/input.txt', 'w') as the_file:
            async for log in client.logs_from(message.channel, limit=1000000000000000):
                try:
                    author = log.author
                except:
                    author = 'invalid'
                messageEncode = str(log.content.encode("utf-8"))[2:-1]

                template = '{message}\n'
                try:
                    the_file.write(template.format(message=messageEncode))
                except:
                    author = log.author.discriminator
                    the_file.write(template.format(message=messageEncode))
        print('Data Collected from ' + message.channel.name)
    elif message.content.startswith('!train'):
        if training != True:
            await client.change_presence(game=None, status='with his brain', afk=False)
            training = True
            tr.train(trainArgs)
        elif training == True:
            await client.change_presence(game=None, status=None, afk=False)
            training = False


client.run(client_secret, bot=True)
