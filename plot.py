import numpy as np
import scipy.io as sio
import sys, os
import matplotlib.pyplot as plt
import networkx as nx

def print_(cov):
	for i in xrange(cov.shape[0]):
		for j in xrange(cov.shape[1]):
			print "%.3f " % cov[i][j],
		print '\n'

def top_k(cov, k):
	a = np.sort(cov, axis=None)
	return a[-k]

def corr(cov, threshold):
	for i in xrange(cov.shape[0]):
		for j in xrange(cov.shape[1]):
			cov[i][j] = cov[i][j] / np.sqrt(cov[i][i]*cov[j][j])
	#print '---', cov.min()
	cov -= cov.min()
	#print_(cov)
	return cov

def plot(cov, fig_name):
	G=nx.Graph()
	# add node
	edgewidth = []
	edgecolor = []
	for i in xrange(cov.shape[0]):
		G.add_node(i)
	# add edge
	for i in xrange(0,cov.shape[0]-1):
		for j in xrange(i+1,cov.shape[0]):
			#print i,j,"%.3f " %cov[i][j],', ',
			G.add_edge(i,j)
			edgewidth.append(cov[i][j])
			edgecolor.append(cov[i][j])
		#print

	#print len(edgewidth), len(edgecolor), len(nx.edges(G))
	
	max_color = 5
	mc = max(edgecolor)
	max_width = 5
	mw = max(edgewidth)
	for i in xrange(len(edgecolor)):
		edgecolor[i] = (edgecolor[i])*max_color/mc
		edgewidth[i] = (edgewidth[i])*max_width/mw
	
	#print 'width', min(edgewidth), max(edgewidth), 'color', min(edgecolor), max(edgecolor)
	# set
	plt.clf()
	#plt.figure(1, size=(4,4))
	pos=nx.spring_layout(G)
	# draw
	nx.draw_networkx_nodes(G, pos, node_size=500, node_color='w', labels=True)
	nx.draw_networkx_labels(G, pos, labels=None, 
									font_size=12, 
									font_color='k', 
									font_family='sans-serif', 
									font_weight='normal', 
									alpha=1.0)
	nx.draw_networkx_edges(G, pos, edge_cmap = plt.cm.Blues,
								   edge_color = edgecolor,
								   alpha = 1,
								   width = edgewidth)
	frame1 = plt.gca()
	frame1.axes.get_xaxis().set_visible(False)
	frame1.axes.get_yaxis().set_visible(False)
	plt.savefig(fig_name+".pdf", format='pdf')

if __name__ == "__main__":
    cov_file = sys.argv[1]
    covs = sio.loadmat(cov_file)
    threshold = 0
    mem1_test = corr(covs['mem1_test'], threshold)
    mem2_test = corr(covs['mem2_test'], threshold)
    rawdata_test = corr(covs['rawdata_test'], threshold)
    plot(mem1_test, 'results/mem1_test')
    plot(mem2_test, 'results/mem2_test')
    plot(rawdata_test, 'results/rawdata_test')
    #mem1_train = corr(covs['mem1_train'], threshold)
    #mem2_train = corr(covs['mem2_train'], threshold)
    #rawdata_train = corr(covs['rawdata_train'], threshold)
    #plot(mem1_train, 'mem1_train')
    #plot(mem2_train, 'mem2_train')
    #plot(rawdata_train, 'rawdata_train')