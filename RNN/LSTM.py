from matplotlib import pyplot as plt
from torch import nn
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

class L_Layer_LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, layers:int):
		super(L_Layer_LSTM, self).__init__()

		self.layers = layers
		self.hidden_size = hidden_size
		self.output_size = output_size

		if layers == 1:
			self.LSTMs = nn.ModuleList([LSTM(input_size, output_size)])
		else:
			self.LSTMs = nn.ModuleList([LSTM(input_size, hidden_size)] +
						  [LSTM(hidden_size, hidden_size) for _ in range(layers-2)] +
						  [LSTM(hidden_size, output_size)])

	def forward(self, x):
		if x.ndim == 1: # If missing B and C
			x = x.unsqueeze(0)
		if x.ndim == 2: # If missing C
			x = x.unsqueeze(2)

		inputs = x
		final_hidden = None
		for lstm in self.LSTMs:
			final_hidden, tmp_inputs = lstm(inputs)
			inputs = torch.stack(tmp_inputs, dim=1)

		return final_hidden, None

class LSTM(nn.Module):
	# (B,1)
	class ForgetGate(nn.Module):
		def __init__(self, input_size, memory_size):
			super(LSTM.ForgetGate, self).__init__()

			self.short_term_memory_weight = nn.Parameter(torch.randn(memory_size, memory_size) * 0.1)
			self.input_weight = nn.Parameter(torch.randn(input_size, memory_size) * 0.1)
			self.input_bias = nn.Parameter(torch.randn(memory_size) * 0.1)

			self.sigmoid = nn.Sigmoid()

		def forward(self, short_term, x):
			return self.sigmoid(short_term @ self.short_term_memory_weight + x @ self.input_weight + self.input_bias)

	# (B,Output)
	class InputGate(nn.Module):
		def __init__(self, input_size, memory_size):
			super(LSTM.InputGate, self).__init__()

			self.short_term_memory_weight_pot = nn.Parameter(torch.randn(memory_size, memory_size) * 0.1)
			self.input_weight_pot = nn.Parameter(torch.randn(input_size, memory_size) * 0.1)
			self.input_bias_pot = nn.Parameter(torch.randn(memory_size) * 0.1)

			self.short_term_memory_weight_perc = nn.Parameter(torch.randn(memory_size, memory_size) * 0.1)
			self.input_weight_perc = nn.Parameter(torch.randn(input_size, memory_size) * 0.1)
			self.input_bias_perc = nn.Parameter(torch.randn(memory_size) * 0.1)

			self.tanh = nn.Tanh()
			self.sigmoid = nn.Sigmoid()

		def forward(self, short_term, x):
			answer = (
					self.sigmoid(
							short_term @ self.short_term_memory_weight_perc +
							x @ self.input_weight_perc +
							self.input_bias_perc) *
					self.tanh(
							short_term @ self.short_term_memory_weight_pot +
							x @ self.input_weight_pot +
							self.input_bias_pot)
			)
			return answer

	# (B, Output)
	class OutputGate(nn.Module):
		def __init__(self, input_size, memory_size):
			super(LSTM.OutputGate, self).__init__()

			self.input_weight = nn.Parameter(torch.randn(input_size, memory_size) * 0.1)
			self.input_bias = nn.Parameter(torch.randn(memory_size) * 0.1)
			self.short_term_memory_weight = nn.Parameter(torch.randn(memory_size, memory_size) * 0.1)

			self.tanh = nn.Tanh()
			self.sigmoid = nn.Sigmoid()

		def forward(self, short_term, x, long_term):
			new_short_term = (
					self.tanh(long_term) *
					self.sigmoid(
							short_term @ self.short_term_memory_weight +
							x @ self.input_weight +
							self.input_bias
					)
			)
			return new_short_term

	def __init__(self, input_size, output_size):
		super(LSTM, self).__init__()
		self.output_size = output_size

		self.forget_gate = LSTM.ForgetGate(input_size, output_size)
		self.create_gate = LSTM.InputGate(input_size, output_size)
		self.output_gate = LSTM.OutputGate(input_size, output_size)

	def forward(self, x):
		if x.ndim == 1:
			x = x.unsqueeze(0)
		if x.ndim == 2:
			x = x.unsqueeze(2)

		long_term_memory = torch.zeros((x.shape[0], self.output_size))
		short_term_memory = torch.zeros((x.shape[0], self.output_size))

		outputs = []

		for timestep in range(0, x.shape[1]):
			cur_timestep = x[:, timestep]
			forget_factor = self.forget_gate(short_term_memory, cur_timestep)
			create_factor = self.create_gate(short_term_memory, cur_timestep)

			# print(long_term_memory, short_term_memory, cur_timestep, forget_factor, create_factor, sep='\n')
			# print(long_term_memory.shape, short_term_memory.shape, cur_timestep.shape, forget_factor.shape, create_factor.shape, sep='\n')
			# print()

			long_term_memory = forget_factor * long_term_memory + create_factor
			short_term_memory = self.output_gate(short_term_memory, cur_timestep, long_term_memory)

			outputs.append(short_term_memory)

		return short_term_memory, outputs


def test_model(model, data, true_answer, label=""):
	optimizer = Adam(model.parameters(), lr=1e-3)
	scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

	criterion = MSELoss()
	losses = []
	for i in range(1000):
		short, _ = model(torch.tensor(data, dtype=torch.float))
		loss = criterion(short.squeeze(1), torch.tensor(true_answer, dtype=torch.float))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step(loss.item())
		losses.append(loss.item())

	final_answer = model(torch.tensor(data, dtype=torch.float))
	print(f"Done with ${label}: Final answer ${str(final_answer[0].squeeze(1).tolist())}")
	plt.plot(losses, label=label)

model_lstm = LSTM(1,1)
model_l_layer = L_Layer_LSTM(1,8,1,3)

x = [
	[1, 0.5, 0.25, 1],
	[0, 0.5, 0.25, 1],
	[0, 1, 0.5, 1],
	[1,0.75,0.5,0.25]
]
y = [1,0, 0.5,1]

test_model(model_l_layer, x, y, label="Layered LSTM")
test_model(model_lstm, x,y, label="LSTM")

plt.legend()
plt.show()