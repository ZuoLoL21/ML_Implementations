from matplotlib import pyplot as plt
from torch import nn
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LSTM(nn.Module):
	def __init__(self, output_size, input_size):
		super(LSTM, self).__init__()
		self.output_size = output_size

		self.forget_gate = ForgetGate(input_size, output_size)
		self.create_gate = InputGate(input_size, output_size)
		self.output_gate = OutputGate(input_size, output_size)

	def forward(self, x):
		if x.ndim == 1:
			x = x.unsqueeze(0)
		if x.ndim == 2:
			x = x.unsqueeze(2)

		long_term_memory = torch.zeros((x.shape[0], self.output_size))
		short_term_memory = torch.zeros((x.shape[0], self.output_size))

		for timestep in range(0, x.shape[1]):
			cur_timestep = x[:, timestep]

			forget_factor = self.forget_gate(short_term_memory, cur_timestep)
			create_factor = self.create_gate(short_term_memory, cur_timestep)

			# print(long_term_memory, short_term_memory, cur_timestep, forget_factor, create_factor, sep='\n')
			# print(long_term_memory.shape, short_term_memory.shape, cur_timestep.shape, forget_factor.shape, create_factor.shape, sep='\n')
			# print()

			long_term_memory = forget_factor * long_term_memory + create_factor

			short_term_memory = self.output_gate(short_term_memory, cur_timestep, long_term_memory)

		return short_term_memory, long_term_memory

# (B,1)
class ForgetGate(nn.Module):
	def __init__(self, input_size, memory_size):
		super(ForgetGate, self).__init__()

		self.short_term_memory_weight = nn.Parameter(torch.randn(memory_size, memory_size) * 0.1)
		self.input_weight = nn.Parameter(torch.randn(input_size, memory_size) * 0.1)
		self.input_bias = nn.Parameter(torch.randn(memory_size) * 0.1)

		self.sigmoid = nn.Sigmoid()

	def forward(self, short_term, x):
		return self.sigmoid(short_term @ self.short_term_memory_weight + x @ self.input_weight + self.input_bias)

# (B,Output)
class InputGate(nn.Module):
	def __init__(self, input_size, memory_size):
		super(InputGate, self).__init__()

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
		super(OutputGate, self).__init__()

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


model = LSTM(1,1)
data = [
	[1, 0.5, 0.25, 1],
	[0, 0.5, 0.25, 1],
]
true_answer = [1,0]

optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

criterion = MSELoss()
losses = []
for i in range(10000):
	short,_ = model(torch.tensor(data, dtype=torch.float))
	loss = criterion(short.squeeze(1), torch.tensor(true_answer, dtype=torch.float))
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	scheduler.step(loss.item())
	losses.append(loss.item())

final_answer = model(torch.tensor(data, dtype=torch.float))
print(final_answer[0].squeeze(1), final_answer[1].squeeze(1), sep='\n')
plt.plot(losses)
plt.show()

