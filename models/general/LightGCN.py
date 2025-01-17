import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from models.BaseModel import GeneralModel, BaseModel

class LightGCNBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--n_layers', type=int, default=3,
							help='Number of LightGCN layers.')
		#ahns参数
		parser.add_argument('--ahns_gamma', type=float, default=0.1, help="Gamma parameter for AHNS")    
		parser.add_argument('--ahns_alpha', type=float, default=0.1, help="Alpha parameter for AHNS")    
		parser.add_argument('--ahns_beta', type=float, default=0.1, help="Beta parameter for AHNS")    
		parser.add_argument('--ahns_p', type=float, default=1, help="P parameter for AHNS")    
		parser.add_argument('--ahns_K', type=int, default=10, help="Number of negative samples in AHNS")    
		parser.add_argument('--test_all', type=int, default=0,
							help='Whether testing on all the items.')
		parser.add_argument('--model_path', type=str, default='',)
			# namespace object has no attribute buffer
		parser.add_argument('--buffer', type=int, default=1000,
						help='Whether to buffer feed dicts for dev/test')
		parser.add_argument('--num_neg', type=int, default=1,
						help='The number of negative items during training.')
		parser.add_argument('--dropout', type=float, default=0,
						help='Dropout probability for each deep layer')
		return parser
	 
	@staticmethod
	def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
		R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
		for user in train_mat:
			for item in train_mat[user]:
				R[user, item] = 1
		R = R.tolil()

		adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
		adj_mat = adj_mat.tolil()

		adj_mat[:user_count, user_count:] = R
		adj_mat[user_count:, :user_count] = R.T
		adj_mat = adj_mat.todok()

		def normalized_adj_single(adj):
			# D^-1/2 * A * D^-1/2
			rowsum = np.array(adj.sum(1)) + 1e-10

			d_inv_sqrt = np.power(rowsum, -0.5).flatten()
			d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
			d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

			bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
			return bi_lap.tocoo()

		if selfloop_flag:
			norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		else:
			norm_adj_mat = normalized_adj_single(adj_mat)

		return norm_adj_mat.tocsr()

	def _base_init(self, args, corpus):
		self.emb_size = args.emb_size
		self.n_layers = args.n_layers
		self.ahns_gamma = args.ahns_gamma
		self.ahns_alpha = args.ahns_alpha
		self.ahns_beta = args.ahns_beta
		self.ahns_p = args.ahns_p
		self.ahns_K = args.ahns_K
		self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
		self._base_define_params()
		self.apply(self.init_weights)
		
	def _base_define_params(self):    
		self.encoder = LightGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers)

	def forward(self, feed_dict):
		
		self.check_list = []
		user = feed_dict['user_id']  # [batch_size]
		item_ids = feed_dict['item_id']  # [batch_size, num_neg+1]
		pos_items = item_ids[:, 0]  # [batch_size]，正样本物品
		neg_items = item_ids[:, 1:]  # [batch_size, num_neg]，负样本物品

		# 获取用户嵌入和正样本嵌入
		user_embed, pos_item_embed = self.encoder(user, pos_items)  # [batch_size, emb_size] 和 [batch_size, emb_size]

		# 确保 pos_item_embed 的形状为 [batch_size, emb_size]，如果不是，进行调整
		if pos_item_embed.ndimension() == 1:
			pos_item_embed = pos_item_embed.unsqueeze(1)  # 如果维度是 [batch_size]，加一维变为 [batch_size, 1, emb_size]

		# 获取负样本的嵌入，这里使用AHNS采样
		neg_item_embed = []
		for k in range(self.ahns_K):
			if neg_items.shape[1] > k:  # 确保k在neg_items的维度范围内
				neg_item = neg_items[:, k].unsqueeze(1)  # 确保neg_item是正确的形状
				neg_item_embed_k = self.adaptive_negative_sampling(user_embed, pos_item_embed, neg_item)
				neg_item_embed.append(neg_item_embed_k)

		# 检查neg_item_embed是否为空，如果不为空，则堆叠负样本嵌入
		if len(neg_item_embed) > 0:
			neg_item_embed = torch.stack(neg_item_embed, dim=1)  # 堆叠负样本嵌入，形状为 [batch_size, K, emb_size]
		else:
			# 如果neg_item_embed为空，则创建一个与pos_item_embed形状相同的零张量
			neg_item_embed = torch.zeros_like(pos_item_embed).unsqueeze(1)

		# 计算BPR损失
		loss = self.create_bpr_loss(user, user_embed, pos_item_embed, neg_item_embed)
		
		# 返回用户嵌入和损失
		return {'prediction': user_embed, 'loss': loss}

		# self.check_list = []
		# user = feed_dict['user_id']#格式为[batch_size]
		# item_ids = feed_dict['item_id']#格式为[batch_size, num_neg+1]
		# pos_items = item_ids[:, 0]#格式为[batch_size]
		# neg_items = item_ids[:, 1:]#格式为[batch_size, num_neg]
		# if neg_items.ndim == 1:
		# 	neg_items = neg_items[:, None]
		# print(f"neg_items shape: {neg_items.shape}")  # 调试输出

		# # 检查neg_items是否为空
		# if neg_items.ndim == 1 or neg_items.shape[1] == 0:
		# 	neg_items = neg_items[:, None]  # 确保至少有一个维度
		# 	neg_items = np.random.randint(0, self.item_num, size=(user.shape[0], self.ahns_K))  # 使用随机负样本填充

		# #获取用户嵌入和正样本嵌入
		# user_embed, pos_item_embed = self.encoder(user, pos_items)#格式为[batch_size, emb_size]
		# #获取负样本的嵌入，这里使用AHNS采样
		# neg_item_embed = []
		# for k in range(self.ahns_K):
		# 	neg_item_embed.append(self.adaptive_negative_sampling(user_embed, pos_item_embed, neg_items[:, k]))

		# neg_item_embed = torch.stack(neg_item_embed, dim=1)## 堆叠负样本嵌入，形状为 [batch_size, K, emb_size]
		
		# loss = self.create_bpr_loss(user, user_embed, pos_item_embed, neg_item_embed)
		# #返回用户嵌入和损失
		# return {'prediction': user_embed, 'loss': loss}

	def create_bpr_loss(self, user, user_embed, pos_item_embed, neg_item_embed):
		"""
		计算 BPR (Bayesian Personalized Ranking) 损失。
		"""
		# 确保 pos_item_embed 的维度和 user_embed 匹配
		# print(f"bpruser_embed shape: {user_embed.shape}")  # 调试输出
		# print(f"bprpos_item_embed shape: {pos_item_embed.shape}")  # 调试输出
		# print(f"bprneg_item_embed shape: {neg_item_embed.shape}")  # 调试输出
		# # 选择每个用户的第一个正样本和第一个负样本项
		pos_item_embed = pos_item_embed[:, 0, :]  # shape: [256, 64]
		neg_item_embed = neg_item_embed[:, 0, :]  # shape: [256, 64]
		# 计算正样本和负样本的预测值

		user_embed = user_embed.unsqueeze(1)
		pos_item_embed = pos_item_embed.unsqueeze(1) 
		if neg_item_embed.dim() == 2 and neg_item_embed.size(1) == 64:
			# 如果 neg_item_embed 的维度是 [256, 64]，则进行拓展
			neg_item_embed = neg_item_embed.unsqueeze(1)  # [256, 1, 64]
		print('neg_item_embed',neg_item_embed.shape)
		print('pos_item_embed',pos_item_embed.shape)
		print('user_embed',user_embed.shape)

		pos_pred = (user_embed * pos_item_embed).sum(dim=-1)  # 计算正样本的预测值
		neg_pred = (user_embed * neg_item_embed).sum(dim=-1)  # 计算负样本的预测值

		# 计算BPR损失
		loss = -torch.log(torch.sigmoid(pos_pred - neg_pred)).mean()
		return loss
		# # 计算正样本和负样本的预测值
		# pos_pred = (user_embed * pos_item_embed).sum(dim=-1)  # 计算正样本的预测值
		# neg_pred = (user_embed[:, None, :] * neg_item_embed).sum(dim=-1)  # 计算负样本的预测值，形状为 [batch_size, K]

		# # BPR损失: log(σ(pred_pos - pred_neg))
		# loss = -torch.log(torch.sigmoid(pos_pred - neg_pred)).mean()

		# return loss
	
	def similarity(self, user_embeddings, item_embeddings): #计算相似度
		"""
		计算用户和物品之间的相似度
		:param user_embeddings: 用户嵌入，形状为 [batch_size, emb_size]
		:param item_embeddings: 物品嵌入，形状为 [batch_size, emb_size]
		:return: 返回用户和物品之间的相似度，形状为 [batch_size]
		"""
		print(user_embeddings.shape)  # 查看 user_embeddings 的形状
		print(item_embeddings.shape)  # 查看 item_embeddings 的形状
		# 假设 item_embeddings 是 [256, 2, 64]

		return (user_embeddings * item_embeddings).sum(dim=-1)

	def adaptive_negative_sampling(self, user_embed, pos_item_embed, neg_item):
		"""
		自适应负样本采样函数
		:param user_embed: 当前批次用户的嵌入，形状为 [batch_size, emb_size]
		:param pos_item_embed: 当前批次正样本物品的嵌入，形状为 [batch_size, emb_size]
		:param neg_item: 当前批次负样本物品的索引，形状为 [batch_size, n_negs]
		:return: 返回选中的负样本的嵌入，形状为 [batch_size, emb_size]
		"""
		batch_size = user_embed.shape[0]
    
		# 检查neg_item是否是张量，如果不是，则将其转换为张量
		if not isinstance(neg_item, torch.Tensor):
			neg_item = torch.tensor(neg_item)
		
		# 确保neg_item至少有两个维度
		if neg_item.ndim < 2:
			neg_item = neg_item.unsqueeze(1)  # 增加一个维度

		n_negs = neg_item.shape[1]  # 负样本的数量

		s_e = user_embed
		p_e = pos_item_embed
		n_e = self.encoder.embedding_dict['item_emb'][neg_item]  # 获取负样本嵌入

		# 对不同hop取平均
		s_e = self.pooling(s_e)
		n_e = self.pooling(n_e)
		p_e = self.pooling(p_e)
		
		s_e = s_e.unsqueeze(1).unsqueeze(2)  # 将 s_e 的形状变为 [256, 1, 1]
		s_e = s_e.repeat(1, 2, 64)  # 扩展为 [256, 2, 64]
		n_e = n_e[:, 0, 0, :] 
		n_e = n_e.unsqueeze(1)  # 将 n_e 的形状变为 [256, 1, 64]
		# print("se2:",s_e.shape)  # [batch_size, n_negs, emb_size]
		# print("pe:",p_e.shape)  # [batch_size, n_negs, emb_size]
		# print("ne:",n_e.shape)  # [batch_size, n_negs, emb_size]

		# 计算相似度
		p_scores = self.similarity(s_e, p_e)
		n_scores = self.similarity(s_e, n_e)

		# 计算相似度差异，三个超参数调整相似度差异的放大程度
		scores = torch.abs(n_scores - self.ahns_beta * (p_scores + self.ahns_alpha).pow(self.ahns_p + 1))

		indices = torch.min(scores, dim=1)[1]  # 选取相似度差异最小的负样本
		indices = indices.clamp(min=0, max=n_e.size(1) - 1)  # 将 indices 限制在合法范围内
		neg_item_embed = n_e[torch.arange(batch_size), indices]
		return neg_item_embed

		# batch_size = user_embed.shape[0]
		# n_negs = neg_item.shape[1]  # 负样本的数量
		# s_e=user_embed
		# p_e=pos_item_embed
		# n_e=self.encoder.embedding_dict['item_emb'][neg_item]
		# #对不同hop取平均
		# s_e = self.pooling(s_e)
		# n_e = self.pooling(n_e)
		# p_e = self.pooling(p_e)
		# #计算相似度
		# p_scores = self.similarity(s_e, p_e)
		# n_scores = self.similarity(s_e, n_e)
		# #计算相似度差异，三个超参数调整相似度差异的放大程度
		# scores = torch.abs(n_scores - self.ahns_beta*(p_scores + self.ahns_alpha).pow(self.ahns_p+1))
		
		# indices = torch.min(scores, dim=1)[1]  # 选取相似度差异最小的负样本
		# neg_item_embed = n_e[torch.arange(batch_size), indices]
		# return neg_item_embed
    
	
	def pooling(self, embeddings): #池化 把多个向量合并成一个向量，为了提取有用特征
	    # [-1, n_hops, channel]
	    return embeddings.mean(dim=1)

    
		
		

class LightGCN(GeneralModel, LightGCNBase):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size','ahns_gamma','ahns_alpha','ahns_beta','ahns_p','ahns_K']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCNBase.parse_model_args(parser)
		return parser

	def __init__(self, args, corpus):
		GeneralModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		out_dict = LightGCNBase.forward(self, feed_dict)
		return {'prediction': out_dict['prediction']}
	
	def loss(self, out_dict):
		predictions = out_dict['prediction']
		pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
		#计算负样本的softmax权重
		neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
		#bpr损失计算
		loss = -(((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1)).clamp(min=1e-8,max=1-1e-8).log().mean()
		return loss
    
	class Dataset(BaseModel.Dataset):
		def _get_feed_dict(self, index):
			def sampling(user_id, train_set, n):
				neg_items = []
				if isinstance(user_id, int):
					user_id = [user_id]
				for user in user_id:
					user = int(user)
					neg_list = []
					for i in range(n):
						while True:
							negitem = np.random.choice(range(self.corpus.n_items))
							if negitem not in train_set.get(user, []):
							
								neg_list.append(negitem)
								break
					if len(neg_list) == 0:  # 如果没有负样本，设置默认值
						neg_list = [np.random.choice(range(self.corpus.n_items))]
					neg_items.append(neg_list)
				return neg_items

			user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
			
			# 判断是训练阶段还是测试阶段
			if self.phase != 'train' and self.model.test_all:
				# 测试阶段，负样本是所有物品（包括目标物品）
				neg_items = np.arange(1, self.corpus.n_items)
			else:
				# 训练阶段，执行自适应负样本采样
				train_set = self.corpus.train_clicked_set  # 获取训练集
				neg_items = sampling(user_id, train_set, self.model.num_neg)  # 采样负样本

			# 确保 neg_items 至少有一个负样本，避免为空
			if len(neg_items) == 0:
				neg_items = [[np.random.choice(range(self.corpus.n_items))]]  # 默认的负样本
			
			# 拼接目标物品和负样本
			target_item = np.array([[target_item]])
			neg_items = np.array(neg_items)
			
			# 确保负样本维度正确
			if neg_items.ndim == 1:
				neg_items = np.expand_dims(neg_items, axis=-1)

			# 拼接 item_ids，确保维度正确
			item_ids = np.concatenate([target_item, neg_items], axis=1).astype(int)

			# 构建 feed_dict
			feed_dict = {
				'user_id': user_id,
				'item_id': item_ids
			}

			return feed_dict
			
		# # Sample negative items for all the instances
		# def actions_before_epoch(self):
		# 	neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))
		# 	for i, u in enumerate(self.data['user_id']):
		# 		clicked_set = self.corpus.train_clicked_set[u]  # neg items are possible to appear in dev/test set
		# 		# clicked_set = self.corpus.clicked_set[u]  # neg items will not include dev/test set
		# 		for j in range(self.model.num_neg):
		# 			while neg_items[i][j] in clicked_set:
		# 				neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
		# 	self.data['neg_items'] = neg_items
			


class LightGCNEncoder(nn.Module):#生成嵌入图
	def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3):
		super(LightGCNEncoder, self).__init__()
		self.user_count = user_count
		self.item_count = item_count
		self.emb_size = emb_size
		self.layers = [emb_size] * n_layers
		self.norm_adj = norm_adj

		self.embedding_dict = self._init_model()
		# self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).cuda()
		self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).cpu()
	def _init_model(self):
		initializer = nn.init.xavier_uniform_
		embedding_dict = nn.ParameterDict({
			'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
			'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
		})
		return embedding_dict

	@staticmethod
	def _convert_sp_mat_to_sp_tensor(X):
		coo = X.tocoo()
		i = torch.LongTensor([coo.row, coo.col])
		v = torch.from_numpy(coo.data).float()
		return torch.sparse.FloatTensor(i, v, coo.shape)

	def forward(self, users, items):
		ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
		all_embeddings = [ego_embeddings]

		for k in range(len(self.layers)):
			ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
			all_embeddings += [ego_embeddings]

		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = torch.mean(all_embeddings, dim=1)

		user_all_embeddings = all_embeddings[:self.user_count, :]
		item_all_embeddings = all_embeddings[self.user_count:, :]

		user_embeddings = user_all_embeddings[users, :]
		item_embeddings = item_all_embeddings[items, :]

		return user_embeddings, item_embeddings
	

    	

# class GeneralModel(BaseModel):
# 	reader, runner = 'BaseReader', 'BaseRunner'

# 	@staticmethod
# 	def parse_model_args(parser):
# 		parser.add_argument('--num_neg', type=int, default=1,
# 							help='The number of negative items during training.')
# 		parser.add_argument('--dropout', type=float, default=0,
# 							help='Dropout probability for each deep layer')
# 		parser.add_argument('--test_all', type=int, default=0,
# 							help='Whether testing on all the items.')
# 		return BaseModel.parse_model_args(parser)

# 	def __init__(self, args, corpus):
# 		super().__init__(args, corpus)
# 		self.user_num = corpus.n_users
# 		self.item_num = corpus.n_items
# 		self.num_neg = args.num_neg
# 		self.dropout = args.dropout
# 		self.test_all = args.test_all

# 	def loss(self, out_dict: dict) -> torch.Tensor:
# 		"""
# 		BPR ranking loss with optimization on multiple negative samples (a little different now to follow the paper ↓)
# 		"Recurrent neural networks with top-k gains for session-based recommendations"
# 		:param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
# 		:return:
# 		"""
# 		# 这里采用的是BPR（Bayesian Personalized Ranking）损失函数
# 		predictions = out_dict['prediction']
# 		pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
# 		neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
# 		loss = -(((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1)).clamp(min=1e-8,max=1-1e-8).log().mean()
# 		# neg_pred = (neg_pred * neg_softmax).sum(dim=1)
# 		# loss = F.softplus(-(pos_pred - neg_pred)).mean()
# 		# ↑ For numerical stability, use 'softplus(-x)' instead of '-log_sigmoid(x)'
# 		return loss

	

