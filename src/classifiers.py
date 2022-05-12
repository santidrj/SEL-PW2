import numpy as np

class CART:


    def _gini_index(self, groups, classes):
        """
        Compute the Gini index for a given group. The value of the index lies between 0 and 1-1/|classes|.
        The smaller the value the greater separation.

        :param groups:
        :param classes:
        :return:
        """
        gini = 0.0
        n_samples = sum([g.shape[0] for g in groups])
        for g in groups:
            if g.shape[0] > 0:
                score = 1.0
                # score the group based on each class
                for c in classes:
                    p = np.count_nonzero(g[:, -1] == c) / g.shape[0]
                    score -= p ** 2
                # weight the group score by size
                gini += (g.shape[0] / n_samples) * score
        return gini

