# Created by Rolando Kindelan Nuñez at 31-07-21
# correspondence mail rolan2kn@gmail.com 
# Feature: This method perform the link based propagation method.
# Enter feature description here

# Scenario: # Enter scenario name here
# Enter steps here
import time

import numpy as np
import heapq


class LinkBasedPropagation:
    CANDIDATE, VISITED = range(2)
    def __init__(self, tdabc, consider_unlabeled=False):
        self.tdabc = tdabc
        self.consider_unlabeled = False
        self.oracle = {}
        self.priority_queue = []

    def execute3(self, sigma, link, fvalues):
        # we create an empty contribution vector
        result = self.tdabc.empty_contribution_vector()

        # if we get invalid values, then we return an empty contribution vector
        if sigma is None or \
                self.tdabc is None:
            return result

        # We need here the link of sigme to be non empty. So we need to recomputed it in this case.
        # If it still empty then, we can't do anything and return an empty contribution vector
        if link is None or fvalues is None or len(link) == 0 or len(fvalues) == 0:
            link, fvalues = self.tdabc.get_link(sigma)
        if link is None or fvalues is None or len(link) == 0 or len(fvalues) == 0:
            return result

        # We set the current simplex sigma to be a CANDIDATE and we update our oracle.
        # This action will prevent us to process several times the same simplices and their links
        self.oracle.update({str(sigma): LinkBasedPropagation.CANDIDATE})

        # We create a priority queue using simplices on the link of sigma as elements and
        # interpreting the filtration values as priority. With this action we guarantee simplices
        # with lower epsilon values are processed before than simplices with higer filtration values.
        for id, f in enumerate(fvalues):
            heapq.heappush(self.priority_queue, (f, link[id]))
            self.oracle.update({str(link[id]): LinkBasedPropagation.CANDIDATE})

        # We start processing simplices on our priority queue. To add new values, we sum up all filtration values
        # from the root point to the current point to guarantee short traversal paths. We are simulating a level-wise
        # traversal of a virtual general tree, where we always resort nodes according to their path-cost upward to
        # the root node (closeness to the original point).
        # We guarantee that closest labels are found first than farthest labels. Then, the cumulative cost is used to
        # ponderate the label contribution.
        while len(self.priority_queue) > 0:
            current_item = heapq.heappop(self.priority_queue)
            priority, mu = current_item
            self.oracle[str(mu)] = self.VISITED  # we set the state of the current simplex

            _link, _fv = self.tdabc.get_link(mu)  # we propagate to
            _fv = np.array(_fv) + priority
            partial_result = self.tdabc.compute_contributions(_link, _fv)

            # if we not found any contribution we collect those simplices on the link of mu to propagate again
            if sum(partial_result) == 0:
                if len(_link) > 0:
                    for id, f in enumerate(_fv):
                        simplex_key = str(_link[id])
                        if simplex_key not in self.oracle:                        # if it is not CANDIDATE nor VISITED
                            heapq.heappush(self.priority_queue, (f, _link[id]))
                            self.oracle.update({simplex_key: LinkBasedPropagation.CANDIDATE})

                    del _link
                    del _fv
            else:
                result += partial_result
        del self.oracle
        del self.priority_queue
        self.oracle = {}
        self.priority_queue = []

        return result

    def execute(self, sigma, link, fvalues):
        # we create an empty contribution vector
        result = self.tdabc.empty_contribution_vector()

        print("\n INIT propagation")

        # if we get invalid values, then we return an empty contribution vector
        if sigma is None or \
                self.tdabc is None:
            return result

        # We need here the link of sigme to be non empty. So we need to recomputed it in this case.
        # If it still empty then, we can't do anything and return an empty contribution vector
        if link is None or fvalues is None or len(link) == 0 or len(fvalues) == 0:
            link, fvalues = self.tdabc.get_link(sigma)
        if link is None or fvalues is None or len(link) == 0 or len(fvalues) == 0:
            return result

        # We set the current simplex sigma to be a CANDIDATE and we update our oracle.
        # This action will prevent us to process several times the same simplices and their links
        self.oracle.update({str(sigma): LinkBasedPropagation.CANDIDATE})

        # We create a priority queue using simplices on the link of sigma as elements and
        # interpreting the filtration values as priority. With this action we guarantee simplices
        # with lower epsilon values are processed before than simplices with higer filtration values.
        for id, f in enumerate(fvalues):
            heapq.heappush(self.priority_queue, (f, link[id]))
            self.oracle.update({str(link[id]): LinkBasedPropagation.CANDIDATE})

        # We start processing simplices on our priority queue. To add new values, we sum up all filtration values
        # from the root point to the current point to guarantee short traversal paths. We are simulating a level-wise
        # traversal in a virtual general tree, where we always resort nodes according to their path-cost upward to
        # the root node (closeness to the original point).
        # We guarantee that closest labels are found first than farthest labels. Then, the cumulative cost is used to
        # ponderate the label contribution.
        t1 = time.time()
        while len(self.priority_queue) > 0:
            current_item = heapq.heappop(self.priority_queue)
            priority, mu = current_item
            print("iteration: tau={0}, p(tau) = {1}".format(mu, priority))
            self.oracle[str(mu)] = self.VISITED  # we set the state of the current simplex

            _link, _fv = self.tdabc.get_link(mu)  # we propagate to
            _fv = np.array(_fv) + priority
            partial_result = self.tdabc.compute_contributions(_link, _fv)

            # if we not found any contribution we collect those simplices on the link of mu to propagate again
            if sum(partial_result) == 0:
                if len(_link) > 0:
                    for id, f in enumerate(_fv):
                        simplex_key = str(_link[id])
                        if simplex_key not in self.oracle:  # if it is not CANDIDATE nor VISITED
                            heapq.heappush(self.priority_queue, (f, _link[id]))
                            self.oracle.update({simplex_key: LinkBasedPropagation.CANDIDATE})

                    del _link
                    del _fv
            else:
                result += partial_result
        t2 = time.time()
        del self.oracle
        del self.priority_queue
        self.oracle = {}
        self.priority_queue = []
        print("\n END propagation in {0} seconds with result={1}".format((t2 - t1), result))
        return result

    def execute2(self, sigma, link, fvalues):
        # we create an empty contribution vector
        result = self.tdabc.empty_contribution_vector()

        print("\n INIT propagation")

        # if we get invalid values, then we return an empty contribution vector
        if sigma is None or \
                self.tdabc is None:
            return result

        # We need here the link of sigme to be non empty. So we need to recomputed it in this case.
        # If it still empty then, we can't do anything and return an empty contribution vector
        if link is None or fvalues is None or len(link) == 0 or len(fvalues) == 0:
            link, fvalues = self.tdabc.get_link(sigma)
        if link is None or fvalues is None or len(link) == 0 or len(fvalues) == 0:
            return result

        # We set the current simplex sigma to be a CANDIDATE and we update our oracle.
        # This action will prevent us to process several times the same simplices and their links
        self.oracle.update({str(sigma): LinkBasedPropagation.CANDIDATE})

        # We create a priority queue using simplices on the link of sigma as elements and
        # interpreting the filtration values as priority. With this action we guarantee simplices
        # with lower epsilon values are processed before than simplices with higer filtration values.
        for id, f in enumerate(fvalues):
            heapq.heappush(self.priority_queue, (f, link[id]))
            self.oracle.update({str(link[id]): LinkBasedPropagation.CANDIDATE})

        # We start processing simplices on our priority queue. To add new values, we sum up all filtration values
        # from the root point to the current point to guarantee short traversal paths. We are simulating a level-wise
        # traversal of a virtual general tree, where we always resort nodes according to their path-cost upward to
        # the root node (closeness to the original point).
        # We guarantee that closest labels are found first than farthest labels. Then, the cumulative cost is used to
        # ponderate the label contribution.
        t1 = time.time()
        while len(self.priority_queue) > 0:

            current_item = heapq.heappop(self.priority_queue)
            priority, tau = current_item
            print("iteration: tau={0}, p(tau) = {1}".format(tau, priority))
            self.oracle[str(tau)] = self.VISITED  # we set the state of the current simplex

            _link, _fv = self.tdabc.get_link(tau)  # we propagate to
            _fv = np.array(_fv) + priority

            if len(_link) > 0:
                for id, f in enumerate(_fv):
                    mu = _link[id]
                    simplex_key = str(mu)
                    if simplex_key in self.oracle:                        # if it is CANDIDATE or VISITED
                        continue

                    partial_result = self.tdabc.compute_contributions([mu], [f])

                    # if we not found label contributions we add mu on the link of mu to propagate again
                    if sum(partial_result) == 0:
                        heapq.heappush(self.priority_queue, (f, mu))
                        self.oracle.update({simplex_key: LinkBasedPropagation.CANDIDATE})
                    else:
                        result += partial_result
                del _link
                del _fv
        t2 = time.time()
        del self.oracle
        del self.priority_queue
        self.oracle = {}
        self.priority_queue = []
        print("\n END propagation in {0} seconds with result={1}".format((t2-t1), result))
        return result
