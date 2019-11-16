#### Model Free Agent 

'''
Object Classes and Relevant Functions for Actor Critic Agent
Author: Annik Carson 
--  Nov 2019
'''

# =====================================
#           IMPORT MODULES            #
# =====================================
from __future__ import division, print_function

import numpy as np

import torch 
from torch.autograd import Variable
from torch import autograd, optim, nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import namedtuple

import stategen as sg

# =====================================
# CLASSES
# =====================================
# Network Class
class AC_Net(nn.Module):
	'''
	An actor-critic neural network class. Takes sensory inputs and generates a policy and a value estimate.
	'''

	# ================================
	def __init__(self, agent_params, **kwargs):
		input_dimensions  = kwargs.get('input_dimensions', agent_params['input_dims'])
		action_dimensions = kwargs.get('action_dimensions', agent_params['action_dims'])
		batch_size        = kwargs.get('batch_size', 4)
		hidden_types      = kwargs.get('hidden_types', agent_params['hid_types'])
		hidden_dimensions = kwargs.get('hidden_dimensions', agent_params['hid_dims'])
		rfsize            = kwargs.get('rfsize', 4)
		padding           = kwargs.get('padding', 1)
		stride            = kwargs.get('stride', 1)
		
		'''
		def __init__(self, input_dimensions, action_dimensions, 
		batch_size=4, hidden_types=[], hidden_dimensions=[],
		rfsize=4, padding=1, stride=1):
		
		AC_Net(input_dimensions, action_dimensions, hidden_types=[], hidden_dimensions=[])

		Create an actor-critic network class.

		Required arguments:
			- input_dimensions (int): the dimensions of the input space
			- action_dimensions (int): the number of possible actions

		Optional arguments:
			- batch_size (int): the size of the batches (default = 4).
			- hidden_types (list of strings): the type of hidden layers to use, options are 'linear',
											  'lstm', 'gru'. If list is empty no hidden layers are
											  used (default = []).
			- hidden_dimensions (list of ints): the dimensions of the hidden layers. Must be a list of
												equal length to hidden_types (default = []).
			- TODO insert new args
		'''
	   
		# call the super-class init 
		super(AC_Net, self).__init__()

		# store the input dimensions
		self.input_d = input_dimensions
		# determine input type
		if type(input_dimensions) == int:
			assert (hidden_types[0] == 'linear' or hidden_types[0] == 'lstm' or hidden_types[0] == 'gru')
			self.input_type = 'vector'
		elif type(input_dimensions) == tuple:
			assert (hidden_types[0] == 'conv' or hidden_types[0] == 'pool')
			self.input_type = 'frame'

		# store the batch size
		self.batch_size = batch_size

		# check that the correct number of hidden dimensions are specified
		assert len(hidden_types) is len(hidden_dimensions)
		
		# check whether we're using hidden layers
		if not hidden_types:

			self.layers = [input_dimensions,action_dimensions]

			# no hidden layers, only input to output, create the actor and critic layers
			self.output = nn.ModuleList([
				nn.Linear(input_dimensions, action_dimensions), # ACTOR
				nn.Linear(input_dimensions, 1)])				# CRITIC
		else:
			# to store a record of the last hidden states
			self.hx = []
			self.cx = []
		
			# create the hidden layers
			self.hidden = nn.ModuleList()
			for i,htype in enumerate(hidden_types):
				
				# check that the type is an accepted one
				assert htype in ['linear','lstm','gru', 'conv', 'pool']

				# get the input dimensions
				if i is 0:
					input_d  = input_dimensions
				else:
					if hidden_types[i-1] in ['conv','pool'] and not htype in ['conv','pool']:
						input_d = int(np.prod(hidden_dimensions[i-1]))
					else:
						input_d = hidden_dimensions[i-1]

				# get the output dimensions
				if not htype in ['conv','pool']:
					output_d = hidden_dimensions[i]
				elif htype in ['conv','pool']:
					output_d = list((0,0,0))
					if htype is 'conv':
						output_d[0] = int(np.floor((input_d[0] + 2*padding - rfsize)/stride) + 1)
						output_d[1] = int(np.floor((input_d[1] + 2*padding - rfsize)/stride) + 1)
						assert output_d[0] == hidden_dimensions[i][0], (hidden_dimensions[i][0], output_d[0])
						assert output_d[1] == hidden_dimensions[i][1]
						output_d[2] = hidden_dimensions[i][2]
					elif htype is 'pool':
						output_d[0] = int(np.floor((input_d[0] +2*padding - (rfsize-1) -1)/stride  +1 ))
						output_d[1] = int(np.floor((input_d[1] +2*padding - (rfsize-1) -1)/stride  +1 ))
						assert output_d[0] == hidden_dimensions[i][0]
						assert output_d[1] == hidden_dimensions[i][1]
						output_d[2] = hidden_dimensions[i][2]
					output_d = tuple(output_d)
				
				# construct the layer
				if htype is 'linear':
					self.hidden.append(nn.Linear(input_d, output_d))
					self.hx.append(None)
					self.cx.append(None)
				elif htype is 'lstm':
					self.hidden.append(nn.LSTMCell(input_d, output_d))
					self.hx.append(Variable(torch.zeros(self.batch_size,output_d)))
					self.cx.append(Variable(torch.zeros(self.batch_size,output_d)))
				elif htype is 'gru':
					self.hidden.append(nn.GRUCell(input_d, output_d))
					self.hx.append(Variable(torch.zeros(self.batch_size,output_d)))
					self.cx.append(None)
				elif htype is 'conv':
					self.hidden.append(nn.Conv2d(input_d[2],output_d[2],rfsize,padding=padding,stride=stride))
					self.hx.append(None)
					self.cx.append(None)
				elif htype is 'pool':
					self.hidden.append(nn.MaxPool2d(rfsize,padding=padding,stride=stride))
					self.hx.append(None)
					self.cx.append(None)

			# create the actor and critic layers
			self.layers = [input_dimensions]+hidden_dimensions+[action_dimensions]
			self.output = nn.ModuleList([
				nn.Linear(output_d, action_dimensions), #actor
				nn.Linear(output_d, 1)                  #critic
			])
		# store the output dimensions
		self.output_d = output_d

		# to store a record of actions and rewards	
		self.saved_actions = []
		self.rewards = []


	# ================================
	def forward(self, x, temperature=1):
		'''
		forward(x):

		Runs a forward pass through the network to get a policy and value.

		Required arguments:
			- x (torch.Tensor): sensory input to the network, should be of size batch x input_d
		'''

		# check the inputs
		if type(self.input_d) == int:
			assert x.shape[-1] == self.input_d
		elif type(self.input_d) == tuple:
			assert (x.shape[2], x.shape[3], x.shape[1]) == self.input_d
			if not  (isinstance(self.hidden[0],nn.Conv2d) or isinstance(self.hidden[0],nn.MaxPool2d)):
				raise Exception('image to non {} layer'.format(self.hidden[0]))

		# pass the data through each hidden layer
		for i, layer in enumerate(self.hidden):
			# squeeze if last layer was conv/pool and this isn't
			if i > 0:
				if (isinstance(self.hidden[i-1],nn.Conv2d) or isinstance(self.hidden[i-1],nn.MaxPool2d)) and \
				not (isinstance(layer,nn.Conv2d) or isinstance(layer,nn.MaxPool2d)):
					x = x.view(1, -1)
			# run input through the layer depending on type
			if isinstance(layer, nn.Linear):
				x = F.relu(layer(x))
			elif isinstance(layer, nn.LSTMCell):
				x, cx = layer(x, (self.hx[i], self.cx[i]))
				self.hx[i] = x.clone()
				self.cx[i] = cx.clone()
			elif isinstance(layer, nn.GRUCell):
				x = layer(x, self.hx[i])
				self.hx[i] = x.clone()
			elif isinstance(layer, nn.Conv2d):
				x = F.relu(layer(x))
			elif isinstance(layer, nn.MaxPool2d):
				x = layer(x)
		# pass to the output layers
		policy = F.softmax(self.output[0](x), dim=1)
		value  = self.output[1](x)

		return policy, value

	# ===============================
	def reinit_hid(self):
		# to store a record of the last hidden states
		self.hx = []
		self.cx = []
	
		for i, layer in enumerate(self.hidden):
			if isinstance(layer, nn.Linear):
				pass
			elif isinstance(layer, nn.LSTMCell):
				self.hx.append(Variable(torch.zeros(self.batch_size,layer.hidden_size)))
				self.cx.append(Variable(torch.zeros(self.batch_size,layer.hidden_size)))
			elif isinstance(layer, nn.GRUCell):
				self.hx.append(Variable(torch.zeros(self.batch_size,layer.hidden_size)))
				self.cx.append(None)
			elif isinstance(layer, nn.Conv2d):
				pass
			elif isinstance(layer, nn.MaxPool2d):
				pass


# =====================================
#            Functions
# =====================================

# softmax for policy layer
def softmax(x, T=1):
	e_x = np.exp((x - np.max(x))/T)
	return np.round(e_x / e_x.sum(axis=0),8) # only difference

### Calculate correct size of next layer if using conv layers
def conv_output(input_tuple, **kwargs):
	h_in, w_in, channels = input_tuple
	padding = kwargs.get('padding', 1) ## because this is 1 in MF, default 0
	dilation = kwargs.get('dilation', 1) # default 1
	kernel_size = kwargs.get('rfsize', 4 ) # set in MF
	stride = kwargs.get('stride', 1) # set in MF, default 1

	h_out = int(np.floor(((h_in +2*padding - dilation*(kernel_size-1) - 1)/stride)+1))
	w_out = int(np.floor(((w_in +2*padding - dilation*(kernel_size-1) - 1)/stride)+1))

	return (h_out, w_out, channels)

# backward rollout to calculate return
def discount_rwds(r, gamma = 0.99):
	disc_rwds = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		running_add = running_add*gamma + r[t]
		disc_rwds[t] = running_add
	return disc_rwds

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
# select action from policy generated by actor-critic network
def select_action(model,policy_, value_):
	a = Categorical(policy_)
	action = a.sample()
	model.saved_actions.append(SavedAction(a.log_prob(action), value_))
	
	return action.item(), policy_.data[0], value_.item()

# calculate policy and value loss for updating network weights
def finish_trial(model, discount_factor, optimizer, **kwargs):
	'''
	finish_trial(model
	Finishes a given training trial and backpropagates.
	'''

	# set the return to zero
	R = 0
	returns_ = discount_rwds(np.asarray(model.rewards), gamma=discount_factor)
	saved_actions = model.saved_actions
	
	policy_losses = []
	value_losses = []
	
	returns_ = torch.Tensor(returns_)

	for (log_prob, value), r in zip(saved_actions, returns_):
		rpe = r - value.item()
		policy_losses.append(-log_prob * rpe)
		## this is a mess --- fix it
		value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([[r]]))).unsqueeze(-1))
	optimizer.zero_grad()

	p_loss = (torch.cat(policy_losses).sum())
	v_loss = (torch.cat(value_losses).sum())

	total_loss = p_loss + v_loss

	total_loss.backward(retain_graph=False)
	optimizer.step()

	del model.rewards[:]
	del model.saved_actions[:]

	return p_loss, v_loss



