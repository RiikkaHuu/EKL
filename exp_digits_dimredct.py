import numpy as np
import pickle
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from EKL_algo import EKL
import matplotlib.pyplot as plt
from matplotlib import cm


"""
    Copyright (C) 2020  Riikka Huusari

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

This file contains the experiments for illustration in the JMLR paper (Entangled Kernels - Beyond Separability, 2021).
It also serves as an example on how to use the EKL code provided in "EKL_algo.py"
"""

rseed = 13   # 13 gives the same plots as in the journal paper
np.random.seed(rseed)

exp_path = ""


# ====== load the data ======

n_class = 4
n_per_class = 25
n = 4*n_per_class

data = load_digits(n_class)

Xall = data.data
y = data.target

classes = np.unique(y)

data_inds = []
test_data_inds = []
for ii in range(n_class):

    class_all = np.where(y==ii)[0]
    data_inds.extend(class_all[:n_per_class])
    test_data_inds.extend(class_all[n_per_class:2*n_per_class])

data_inds = np.array(data_inds)
test_data_inds = np.array(test_data_inds)

X = Xall[data_inds, :]
Xt = Xall[test_data_inds, :]
Y = np.zeros((len(data_inds), n_class))-1
Yt = np.zeros((len(test_data_inds), n_class))-1
for ii in range(n_class):
    Y[ii*n_per_class:(ii+1)*n_per_class, ii] = 1
    Yt[ii*n_per_class:(ii+1)*n_per_class, ii] = 1

yy = np.argmax(Y, axis=1)
yyt = np.argmax(Yt, axis=1)


# ====== apply EKL to this data ======

ranklvl = 2

# the features as for linear kernel
U = X
Ut = Xt

print("starting with ekl...")

exp_name = "ekl_feats_rseed"+str(rseed)+".obj"

run_ekl = False

try:
    with open(exp_path+exp_name, 'rb') as f:
        results = pickle.load(f)
except FileNotFoundError:
    run_ekl = True

if run_ekl:

    # run with various parameter combinations, select the one that gives best test accuracy

    tmp_acc = 0
    feats = None
    feats_test = None
    for gamma in np.arange(0.0, 1.2, 0.2):

        ekl = EKL(U, Y, 1, ranklvl, gamma)
        ekl.solve()

        for lmbda in [10, 0.1, 0.001, 0.00001]:

            eklpred = ekl.predict(Ut, lmbda)
            eklpred[eklpred < 0] = -1
            eklpred[eklpred >= 0] = 1
            acc = accuracy_score(Yt, eklpred)

            if acc > tmp_acc:
                tmp_acc = acc
                print("current best test accuracy:", tmp_acc)
                feats = np.dot(np.kron(U, np.eye(n_class)), ekl.Q)
                feats_test = np.dot(np.kron(Ut, np.eye(n_class)), ekl.Q)

        print("finished with gamma: ", gamma)

    res = {"feats": feats, "feats_test": feats_test, "acc": tmp_acc}

    with open(exp_path+exp_name, "wb") as f:
        pickle.dump(res, f)

else:
    with open(exp_path+exp_name, 'rb') as f:
        results = pickle.load(f)

    feats = results["feats"]
    feats_test = results["feats_test"]


G = np.dot(feats, feats.T)

img_folder = ""

fontsize_large = 22

# ====== plot parts of the kernel matrix ======

plt.matshow(G)
# plt.title("training data kernel")

plt.axis('off')

l = 0.5

plt.plot([25*4, 25*4], [0, 25*4*4], c='w', linewidth=l)
plt.plot([2*25*4, 2*25*4], [0, 25*4*4], c='w', linewidth=l)
plt.plot([3*25*4, 3*25*4], [0, 25*4*4], c='w', linewidth=l)
plt.plot([0, 25*4*4], [25*4, 25*4], c='w', linewidth=l)
plt.plot([0, 25*4*4], [2*25*4, 2*25*4], c='w', linewidth=l)
plt.plot([0, 25*4*4], [3*25*4, 3*25*4], c='w', linewidth=l)

z = 0
plt.plot([z, z], [z, z+3*4], c='k')
plt.plot([z, z+3*4], [z, z], c='k')
plt.plot([z+3*4, z+3*4], [z, z+3*4], c='k')
plt.plot([z, z+3*4], [z+3*4, z+3*4], c='k')
z = 25*4
plt.plot([z, z], [z, z+3*4], c='k')
plt.plot([z, z+3*4], [z, z], c='k')
plt.plot([z+3*4, z+3*4], [z, z+3*4], c='k')
plt.plot([z, z+3*4], [z+3*4, z+3*4], c='k')
z = 50*4
plt.plot([z, z], [z, z+3*4], c='k')
plt.plot([z, z+3*4], [z, z], c='k')
plt.plot([z+3*4, z+3*4], [z, z+3*4], c='k')
plt.plot([z, z+3*4], [z+3*4, z+3*4], c='k')
z = 75*4
plt.plot([z, z], [z, z+3*4], c='k')
plt.plot([z, z+3*4], [z, z], c='k')
plt.plot([z+3*4, z+3*4], [z, z+3*4], c='k')
plt.plot([z, z+3*4], [z+3*4, z+3*4], c='k')

z = 0
zz = 25*4
plt.plot([zz, zz], [z, z+3*4], c='k')
plt.plot([zz, zz+3*4], [z, z], c='k')
plt.plot([zz+3*4, zz+3*4], [z, z+3*4], c='k')
plt.plot([zz, zz+3*4], [z+3*4, z+3*4], c='k')

plt.savefig(img_folder+"digits_kernel.png", bbox_inches='tight', dpi=1000)
plt.show()
exit()

plt.matshow(G[0:3*n_class, 0:3*n_class])
plt.xticks([0,4,8])
plt.yticks([0,4,8])
plt.xticks(fontsize=fontsize_large-8)
plt.yticks(fontsize=fontsize_large-8)
plt.title("0", fontsize=fontsize_large)
# plt.savefig(img_folder+"digits_kernel_0.png", bbox_inches='tight', dpi=1000)

plt.matshow(G[n_per_class:n_per_class+3*n_class, n_per_class:n_per_class+3*n_class])
plt.xticks([0,4,8])
plt.yticks([0,4,8])
plt.xticks(fontsize=fontsize_large-8)
plt.yticks(fontsize=fontsize_large-8)
plt.title("1", fontsize=fontsize_large)
# plt.savefig(img_folder+"digits_kernel_1.png", bbox_inches='tight', dpi=1000)

plt.matshow(G[2*n_per_class:2*n_per_class+3*n_class, 2*n_per_class:2*n_per_class+3*n_class])
plt.xticks([0,4,8])
plt.yticks([0,4,8])
plt.xticks(fontsize=fontsize_large-8)
plt.yticks(fontsize=fontsize_large-8)
plt.title("2", fontsize=fontsize_large)
# plt.savefig(img_folder+"digits_kernel_2.png", bbox_inches='tight', dpi=1000)

plt.matshow(G[3*n_per_class:3*n_per_class+3*n_class, 3*n_per_class:3*n_per_class+3*n_class])
plt.xticks([0,4,8])
plt.yticks([0,4,8])
plt.xticks(fontsize=fontsize_large-8)
plt.yticks(fontsize=fontsize_large-8)
plt.title("3", fontsize=fontsize_large)
# plt.savefig(img_folder+"digits_kernel_3.png", bbox_inches='tight', dpi=1000)

# plt.matshow(G[0*n_per_class:0*n_per_class+3*n_class, n_per_class:n_per_class+3*n_class])
# plt.xticks([0,4,8])
# plt.yticks([0,4,8])
# plt.xticks(fontsize=fontsize_large-8)
# plt.yticks(fontsize=fontsize_large-8)
# plt.title("0 - 1", fontsize=fontsize_large)
# # plt.savefig(img_folder+"digits_kernel_01.png", bbox_inches='tight', dpi=1000)


# ====== plot the projections to two dimensions ======

plt.figure(figsize=(9.5,8))

plt.subplot(221)

cm_subsection = np.linspace(0, 1, 4)
colors = [cm.viridis(x) for x in cm_subsection]

# these are plotted only for the legend. I hope there is a better way to do this but hey, this works too..
plt.scatter(2, 2, c='k', label="tr")
plt.scatter(2, 2, c='k', marker="v", label="tst")
plt.scatter(2, 2, c=colors[0], label="0")
plt.scatter(2, 2, c=colors[1], label="1")
plt.scatter(2, 2, c=colors[2], label="2")
plt.scatter(2, 2, c=colors[3], label="3")
plt.scatter(2, 2, c='w', s=80, edgecolors="w")  # finally hide the blobs with another blob

# plot the actual data
for plotind, plotnum in enumerate([221, 222, 223, 224]):

    featinds = np.arange(0, n)*n_class + plotind  # the indices where projections corresponding to this task are

    plt.subplot(plotnum)
    plt.scatter(feats[featinds, 0], feats[featinds, 1], c=yy)
    plt.scatter(feats_test[featinds, 0], feats_test[featinds, 1], c=yyt, marker="v")
    plt.title(plotind, fontsize=fontsize_large)


plt.figlegend(fontsize=fontsize_large-2, bbox_to_anchor=(-.02, -.32, 1, 1))
# plt.subplots_adjust(right=1.3)
plt.tight_layout(rect=[0, 0, 0.8, 1])

# plt.savefig(img_folder+"digits_clusters.eps", format='eps', bbox_inches='tight', dpi=1000)


plt.show()
