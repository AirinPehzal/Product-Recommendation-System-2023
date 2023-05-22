import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import itertools
import matplotlib.pyplot as plt

'''
Calculates the cosine simularity between words
'''
cosine_function = lambda a, b: np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def conditional_prob(x, y):
    '''
    Calculates the conditional probability of having x in the document given y and vice versa
    '''
    x_given_y = np.count_nonzero(pd.Series(x * y)) / np.count_nonzero(y)
    y_given_x = np.count_nonzero(pd.Series(x * y)) / np.count_nonzero(x)
    return x_given_y, y_given_x


def weighted_n(x, y):
    '''
    Calculates the weighted similarity between words
    '''
    return len(set(x) & set(y)) / ((len(x) + len(y) / 2))


def L(x, y, docs_words, cooccured):
    '''
    Calculates the L metric for discovering hierarchical links 
    '''
    P_x_y, P_y_x = conditional_prob(docs_words[x], docs_words[y])
    c_x_y = cosine_function(docs_words[x], docs_words[y])
    N_x_y = weighted_n(cooccured[x], cooccured[y])
    return (P_y_x - P_x_y) * c_x_y * (N_x_y + 1)


def find_ind_for_removal(known_edges, all_edges):
    '''
    Finds the index of the word for which we can potentially remove an edge from the graph
    '''
    for i in range(len(all_edges)):
        if all_edges[i][0] == known_edges[0] and all_edges[i][1] == known_edges[1]:
            return i


def s(x, y, docs_words, G1):
    '''
    Calculates the s metric for discovering related/equivalent words
    '''
    cos_sim = cosine_function(docs_words[x], docs_words[y])
    parents = nx.ancestors(G1, x) & nx.ancestors(G1, y)
    permutated_parents = list(itertools.combinations(parents, 2))

    av_sim = []
    sup_sim = 0
    for pair in permutated_parents:
        av_sim.append(cosine_function(docs_words[pair[0]], docs_words[pair[1]]))

    if av_sim != []:
        sup_sim = sum(av_sim) / len(av_sim)

    P_x_y, P_y_x = conditional_prob(docs_words[x], docs_words[y])
    abs_dif = np.abs(P_x_y - P_y_x)

    return cos_sim - 0.2 * sup_sim - 0.2 * abs_dif


def distance(x, y, docs_words, G1):
    '''
    Calculates the distance between words as 1 / s(x, y)
    If the distance is too small, returns a numbers obviously outside of range
    '''
    d = s(x, y, docs_words, G1)
    if d != 0:
        return 1 / d
    else:
        return 1000


def single_linkage_distance(clust_a, clust_b, docs_words, G1):
    '''
    Calculates the single linkage distance for 2 clusters
    '''
    minimum_dist = 100
    a = pd.concat([clust_a, clust_b])

    all_links = list(itertools.combinations(list(a), 2))

    for link in all_links:
        d = distance(link[0], link[1], docs_words, G1)

        if d < minimum_dist:
            minimum_dist = d

    return minimum_dist


def create_ontology(united_columns, min_df=0.01, top_ngram=1, min_cooccur=50):
    '''
    Function for creating the ontology
    '''
    #Initializing the value which checks if the ontology is ready
    check = -1

    while check != 0:
        
        #Initializing TF-IDF with inputed parameters and using it on the input data
        tfidf = TfidfVectorizer(ngram_range=(1, top_ngram), min_df=min_df)
        tf_idf_united = tfidf.fit_transform(united_columns)
        #Results of TF-IDF are saved into docs_words - dataframe which has words as columns and document indexes as rows
        docs_words = pd.DataFrame(tf_idf_united.toarray(), columns=tfidf.get_feature_names_out())

        cooccured = {}
        
        #For every word in docs_words we create a list of cooccured words 
        #which happen in min_occur or more documents where the word is
        for cur_word in enumerate(docs_words.columns):
            current_list = []
            for other_word in enumerate(docs_words.columns):
                if cur_word[0] != other_word[0] and np.count_nonzero(
                        pd.Series(docs_words.iloc[:, cur_word[0]] * docs_words.iloc[:, other_word[0]])) >= min_cooccur:
                    current_list.append(other_word[1])
            cooccured[cur_word[1]] = current_list

        work_nodes = set()
        work_edges = []
        
        #For every word in docs_words and their cooccured words
        #we calculates L metric
        #if it is above the critical value, then we infer a hierarchical link
        for word in docs_words.columns:
            for other_word in cooccured[word]:
                if L(word, other_word, docs_words, cooccured) > 0.2:
                    work_nodes.add(word)
                    work_nodes.add(other_word)
                    work_edges.append(
                        [other_word, word, {'weight': np.round(L(word, other_word, docs_words, cooccured), 4)}])

        just_edges = {}
        
        #Getting a list of all edges in the ontology
        for edge in work_edges:
            if edge[0] not in just_edges:
                just_edges[edge[0]] = [edge[1]]
            else:
                just_edges[edge[0]].append(edge[1])

        edge_for_removal = []
        
        #Finding all edges which make a triangle
        for cur_node in just_edges:
            for other_node in just_edges[cur_node]:
                if other_node in just_edges and set(just_edges[cur_node]) & set(just_edges[other_node]) != set():
                    edge_for_removal.append(
                        [cur_node, other_node, list(set(just_edges[cur_node]) & set(just_edges[other_node]))])
        
        #Removing the weakest edge (= with the least value of L metric)
        for edges in edge_for_removal:
            for member in edges[2]:
                ind1 = find_ind_for_removal([edges[0], member], work_edges)
                ind2 = find_ind_for_removal([edges[1], member], work_edges)

                if ind1 is not None and ind2 is not None:
                    if work_edges[ind1][2]['weight'] >= work_edges[ind2][2]['weight']:
                        work_edges.pop(ind2)
                    else:
                        work_edges.pop(ind1)
        
        #Saving the ontology into a graph structure
        G1 = nx.DiGraph()

        G1.add_nodes_from(work_nodes)
        G1.add_edges_from(work_edges)

        rel_equiv = set()

        visited = set()
        
        #Checking all words in the ontology for possible related/equivalent relationship
        #If s metric for two words is above the critical value
        #We assume that they are related/equivalent
        for word in list(nx.nodes(G1)):
            for other_word in list(nx.nodes(G1)):
                if word != other_word and other_word not in visited:
                    if s(word, other_word, docs_words, G1) >= 0.75:
                        rel_equiv.add(word)
                        rel_equiv.add(other_word)
            visited.add(word)
        
        #Manually conctructed hierarchical clustering algoritm using single linkage = 1/s metric
        clustering = pd.DataFrame(list(rel_equiv), columns=['word'])
        clustering['cluster'] = list(clustering.index)

        curr_clusters = clustering['cluster'].copy()
        visited_clusts = set()

        for clust in curr_clusters:
            visited_clusts.add(clust)
            curr_clust = clustering[clustering['cluster'] == clust]
            other_clust = clustering[clustering['cluster'] != clust]

            for other in other_clust['cluster']:
                if other not in visited_clusts:
                    another = other_clust[other_clust['cluster'] == other]

                    if single_linkage_distance(curr_clust['word'], another['word'], docs_words, G1) <= (1 / 0.75):
                        clustering.loc[clustering['cluster'] == other, ['cluster']] = clust
        
        #Printing the results of clustering to see what was found
        print(clustering)
        
        #Checking if ontology is ready
        #If yes, the algorithm stops and returns the final processed dataset
        check = len(clustering['cluster']) - len(clustering['cluster'].unique())
        
        #If not
        if check != 0:
            
            #For every cluster we create an aggravated word (separator between them is _)
            for cluster in clustering['cluster'].unique():
                curr = clustering[clustering['cluster'] == cluster]
                in_work = curr.word.str.replace(' ', '_')
                new_word_unique = set('_'.join(in_work).split('_'))
                new_word = '_'.join(list(new_word_unique))
                clustering.loc[clustering['cluster'] == cluster, ['replace']] = new_word

            k = united_columns.copy()

            new_united = []
            
            #Processing the original dataset to include the related/equivalent relationships
            for row in k:
                
                #For every cluster we take the list of original words and the aggrevated word
                #for this cluster
                for cluster in clustering['cluster'].unique():
                    words = clustering[clustering['cluster'] == cluster]
                    words = words['word']
                    replacement = clustering[clustering['cluster'] == cluster]
                    replacement = replacement['replace'].iloc[0]
                    
                    #Remove all original words from the document
                    count = 0
                    for w in words:
                        while row.find(w) != -1:
                            count += 1
                            row = row.replace(w, ' ')
                            
                    #If any words were removed, normalize the whitespaces
                    while row.find('  ') != -1:
                        row = row.replace('  ', ' ')
                        
                    #If any words from this cluster were removed, add the aggrevated word at the end of the document
                    if count != 0:
                        row = row + ' ' + replacement

                new_united.append(row)
            
            #Update the dataset and start Klink algorithm again
            united_columns = new_united
    
    #Return the ontology
    return united_columns


def visualise_ontology(united_columns, min_df=0.01, top_ngram=1, min_cooccur=50):
    '''
    Function for visualising the ontology
    '''
    #Everything is the same as in create_ontology
    tfidf = TfidfVectorizer(ngram_range=(1, top_ngram), min_df=min_df)
    tf_idf_united = tfidf.fit_transform(united_columns)
    docs_words = pd.DataFrame(tf_idf_united.toarray(), columns=tfidf.get_feature_names_out())

    cooccured = {}

    for cur_word in enumerate(docs_words.columns):
        current_list = []
        for other_word in enumerate(docs_words.columns):
            if cur_word[0] != other_word[0] and np.count_nonzero(
                    pd.Series(docs_words.iloc[:, cur_word[0]] * docs_words.iloc[:, other_word[0]])) >= min_cooccur:
                current_list.append(other_word[1])
        cooccured[cur_word[1]] = current_list

    work_nodes = set()
    work_edges = []

    for word in docs_words.columns:
        for other_word in cooccured[word]:
            if L(word, other_word, docs_words, cooccured) > 0.2:
                work_nodes.add(word)
                work_nodes.add(other_word)
                work_edges.append(
                    [other_word, word, {'weight': np.round(L(word, other_word, docs_words, cooccured), 4)}])

    just_edges = {}

    for edge in work_edges:
        if edge[0] not in just_edges:
            just_edges[edge[0]] = [edge[1]]
        else:
            just_edges[edge[0]].append(edge[1])

    edge_for_removal = []

    for cur_node in just_edges:
        for other_node in just_edges[cur_node]:
            if other_node in just_edges and set(just_edges[cur_node]) & set(just_edges[other_node]) != set():
                edge_for_removal.append(
                    [cur_node, other_node, list(set(just_edges[cur_node]) & set(just_edges[other_node]))])

    for edges in edge_for_removal:
        for member in edges[2]:
            ind1 = find_ind_for_removal([edges[0], member], work_edges)
            ind2 = find_ind_for_removal([edges[1], member], work_edges)

            if ind1 is not None and ind2 is not None:
                if work_edges[ind1][2]['weight'] >= work_edges[ind2][2]['weight']:
                    work_edges.pop(ind2)
                else:
                    work_edges.pop(ind1)
    
    #Inputing the ontology into a graph structure
    G1 = nx.DiGraph()

    G1.add_nodes_from(work_nodes)
    G1.add_edges_from(work_edges)
    
    #Setting up the parameters of visualisation
    plt.rcParams["figure.figsize"] = (20, 20)

    pos = nx.spring_layout(G1, seed=47)
    node_sizes = [i for i in range(len(G1))]

    nodes = nx.draw_networkx_nodes(G1, pos, node_size=node_sizes, node_color="indigo", alpha=0.75)

    M = G1.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    cmap = plt.cm.plasma

    edges = nx.draw_networkx_edges(
            G1,
            pos,
            node_size=node_sizes,
            arrowstyle="->",
            arrowsize=10,
            edge_color=edge_colors,
            edge_cmap=cmap,
            width=1,
        )

    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])
    
    #Show the visualisation
    nx.draw_networkx_labels(G1, pos, font_size=10, verticalalignment='top')
    plt.show()
