import torch
import os
import random
import utils
import data_utils
import similarity
import argparse
import datetime
import json

from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser(description='Settings for creating CBM')


parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--concept_set", type=str, default=None,
                    help="path to concept set name")
parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")

parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
parser.add_argument("--proj_batch_size", type=int, default=50000, help="Batch size to use when learning projection layer")

parser.add_argument("--feature_layer", type=str, default='layer4',
                    help="Which layer to collect activations from. Should be the name of second to last layer in the model")
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="save location for backbone and CLIP activations")
parser.add_argument("--save_dir", type=str, default='saved_models', help="where to save trained models")
parser.add_argument("--clip_cutoff", type=float, default=0.25, help="concepts with smaller top5 clip activation will be deleted")
parser.add_argument("--proj_steps", type=int, default=1000, help="how many steps to train the projection layer for")
parser.add_argument("--interpretability_cutoff", type=float, default=0.45, help="concepts with smaller similarity to target concept will be deleted")
parser.add_argument("--lam", type=float, default=0.0007, help="Sparsity regularization parameter, higher->more sparse")
parser.add_argument("--n_iters", type=int, default=1000, help="How many iterations to run the final layer solver for")
parser.add_argument("--print", action='store_true', help="Print all concepts being deleted in this stage")

def train_cbm_and_save(args):

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.concept_set==None:
        args.concept_set = "data/concept_sets/{}_filtered.txt".format(args.dataset)

    similarity_fn = similarity.cos_similarity_cubed_single

    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"

    #get concept set
    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")

    with open(args.concept_set) as f:
        concepts = f.read().split("\n")

    #save activations and get save_paths
    for d_probe in [d_train, d_val]:
        utils.save_activations(clip_name = args.clip_name, target_name = args.backbone,
                               target_layers = [args.feature_layer], d_probe = d_probe,
                               concept_set = args.concept_set, batch_size = args.batch_size,
                               device = args.device, pool_mode = "avg", save_dir = args.activation_dir)

    target_save_name, clip_save_name, text_save_name = utils.get_save_names(args.clip_name, args.backbone,
                                            args.feature_layer,d_train, args.concept_set, "avg", args.activation_dir)
    val_target_save_name, val_clip_save_name, text_save_name =  utils.get_save_names(args.clip_name, args.backbone,
                                            args.feature_layer, d_val, args.concept_set, "avg", args.activation_dir)

    #load features
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu").float()

        val_target_features = torch.load(val_target_save_name, map_location="cpu").float()

        image_features = torch.load(clip_save_name, map_location="cpu").float()
        # 计算输入张量的范数。范数可以看作是向量的“长度”或“大小”。/= 是一个原位操作符，它表示将 image_features 中每个元素除以相应范数的结果，从而完成归一化。
        # norm：对每一行计算 L2 范数。
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
        val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        # 这个符号在 Python 中表示 矩阵乘法，可以理解为常用的矩阵点积运算（dot product）。在 PyTorch 中，它与 torch.matmul() 相等价。
        # 计算每个图像嵌入和每个文本嵌入之间的 内积（dot product），得到一个矩阵 clip_features。
        # 结果 clip_features[i, j] 表示第 i 个图像嵌入与第 j 个文本嵌入之间的相似度。
        # 这样做可以得到每个图像与所有文本之间的相似度，或者换句话说，是每个文本与所有图像之间的相似度。
        clip_features = image_features @ text_features.T
        # clip_features 形状是 (n_images, n_texts)，即有 n_images行和 n_texts列
        val_clip_features = val_image_features @ text_features.T

        del image_features, text_features, val_image_features

    #filter concepts not activating highly
    '''为什么取前五？'''
    # torch.topk(clip_features, dim=0, k=5)：按照列，找出每列中前5个最大的值，依次从第一行排到低五行。
    # 然后取前五的列最大值的平均值，按列求平均，返回平均值。
    highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)

    if args.print:
        for i, concept in enumerate(concepts):
            if highest[i]<=args.clip_cutoff:# clip_cutoff=0.25
                print("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i])) # 删除每列激活度低于0.25的概念
    concepts = [concepts[i] for i in range(len(concepts)) if highest[i]>args.clip_cutoff] # 保留下来的激活度高的概念

    #save memory by recalculating
    del clip_features
    with torch.no_grad():

        image_features = torch.load(clip_save_name, map_location="cpu").float() # len_image_feature:50000
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        # highest > args.clip_cutoff 将生成如下布尔张量，形状与 image_features 相同。
        # 其中，布尔张量中的元素为 True 表示对应的图像嵌入的 top5 最大值大于 args.clip_cutoff；
        # 而 False 表示对应的图像嵌入的 top5 最大值小于等于 args.clip_cutoff。
        text_features = torch.load(text_save_name, map_location="cpu").float()[highest>args.clip_cutoff]
        # norm是计算L2范数
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        # 重新计算相似度。因为删除概念之后，归一化已经改变了
        clip_features = image_features @ text_features.T
        del image_features, text_features

    val_clip_features = val_clip_features[:, highest>args.clip_cutoff]

    #learn projection layer：学习投影层W_C

    #in_features: 输入特征的数量。在这里，使用 target_features.shape[1] 表示输入特征的维度，假设 target_features
    # 是一个形状为 (n_samples, feature_dim) 的张量，feature_dim 是输入特征的维度。
    # out_features: 输出特征的数量，这里使用 len(concepts)，表示输出的维度等于概念的数量。
    # bias=False: 表示不使用偏置项。如果不需要偏置项，可以将其设置为 False，这在某些情况下是合适的。
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts),
                                 bias=False).to(args.device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)

    # indices 列表将包含从 0 到 len(target_features) - 1 的所有索引，通常用于访问 target_features 中的每个样本。
    # indices =  [0 - 49999]
    indices = [ind for ind in range(len(target_features))]

    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    for i in range(args.proj_steps):
        # 从 indices 列表中随机抽取 k 个不重复的元素。这里的 k 是 proj_batch_size，表示希望抽取的样本数量。
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))

        # 停止梯度计算：
        # 使用 detach() 后，得到的新张量与原张量共享同样的数据，但不再与计算图相连。
        # 这意味着在后续的操作中，任何对这个新张量的操作都不会影响原张量的梯度计算。
        # target_features.shape:torch.Size([50000, 1024])
        outs = proj_layer(target_features[batch].to(args.device).detach())
        # clip_features 形状是 (n_images, n_texts)，即有 n_images行和 n_texts列
        # clip_features.shape:torch.Size([50000, 138])
        loss = -similarity_fn(clip_features[batch].to(args.device).detach(), outs)


        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i%50==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_loss = -similarity_fn(val_clip_features.to(args.device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if i==0:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(best_step, -loss.cpu(),
                                                                                               -best_val_loss.cpu()))

            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else: #stop if val loss starts increasing
                '''
                改进建议:使用耐心（Patience）：可以设置一个耐心参数（例如，允许连续多次验证损失没有改善后再停止），这样可以避免因为一次波动而提前结束训练。
                patience = 5  # 允许的连续不改善次数
                patience_counter = 0
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = i
                    best_weights = proj_layer.weight.clone()
                    patience_counter = 0  # 重置耐心计数
                else:
                    patience_counter += 1
                    if patience_counter > patience:
                        break  # 触发早停
                '''
                break
        opt.zero_grad()

    proj_layer.load_state_dict({"weight":best_weights})
    print("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))

    #delete concepts that are not interpretable(其实是学不会的)
    with torch.no_grad():
        outs = proj_layer(val_target_features.to(args.device).detach())
        # sim输出的是每个概念的相似度，形状是 (len(concepts),)
        sim = similarity_fn(val_clip_features.to(args.device).detach(), outs)
        interpretable = sim > args.interpretability_cutoff

    if args.print:
        for i, concept in enumerate(concepts):
            if sim[i]<=args.interpretability_cutoff:
                print("Deleting {}, Iterpretability:{:.3f}".format(concept, sim[i]))

    concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]

    del clip_features, val_clip_features

    # proj_layer对不可解释的概念进行删除（列）
    W_c = proj_layer.weight[interpretable]
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight":W_c})

    train_targets = data_utils.get_targets_only(d_train)
    val_targets = data_utils.get_targets_only(d_val)

    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())

        # 计算均值和标准差:
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)

        # 中心化：减去均值，使得数据的均值为0。
        train_c -= train_mean
        # 缩放：将数据除以标准差，使得数据的标准差为1。
        train_c /= train_std

        train_y = torch.LongTensor(train_targets)
        indexed_train_ds = IndexedTensorDataset(train_c, train_y)


        val_c -= train_mean
        val_c /= train_std

        val_y = torch.LongTensor(val_targets)
        val_ds = TensorDataset(val_c,val_y)


    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    # Make linear model and zero initialize
    linear = torch.nn.Linear(train_c.shape[1],len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    # Solve the GLM path
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
                      val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = len(classes))
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']

    save_name = "{}/{}_cbm_{}".format(args.save_dir, args.dataset, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    os.mkdir(save_name)
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))

    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)

    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        out_dict = {}
        for key in ('lam', 'lr', 'alpha', 'time'):
            out_dict[key] = float(output_proj['path'][0][key])
        out_dict['metrics'] = output_proj['path'][0]['metrics']
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
        json.dump(out_dict, f, indent=2)

if __name__=='__main__':
    args = parser.parse_args()
    train_cbm_and_save(args)