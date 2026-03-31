import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors


class IsUMap:
    def __init__(
        self,
        n_neighbors: int = 15,
        n_components: int = 2,
        metric: str = 'euclidean',
        mode: str = 'um',
        use_rho: bool = True,
        refine_metric_mds: bool = False,
        mds_max_iter: int = 300,
        mds_n_init: int = 1,
        random_state: int | None = 42
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.mode = mode
        self.use_rho = use_rho
        self.refine_metric_mds = refine_metric_mds
        self.mds_max_iter = mds_max_iter
        self.mds_n_init = mds_n_init
        self.random_state = random_state
    
    def _validate_params(self, n_samples: int):
        """Validate the parameters"""
        if not isinstance(self.n_neighbors, int) or self.n_neighbors < 2:
            raise ValueError('n_neighbors must be an integer >= 2')
        
        if self.n_neighbors >= n_samples:
            raise ValueError('n_neighbors must be < n_samples')
        
        if not isinstance(self.n_components, int) or self.n_components < 1:
            raise ValueError('n_components must be an integer >= 1')
        
        if self.mode not in ('epmet', 'um'):
            raise ValueError('mode must be epmet or um')
    
    @staticmethod
    def update_edge(edge_weights: dict[tuple[int, int], float], u: int, v: int, w: float):
        """Update edge weight"""
        if u != v:
            a, b = (u, v) if u < v else (v, u)
            old = edge_weights.get((a, b))
            if old is None or w < old:
                edge_weights[(a, b)] = w

    def build_merged_graph(self, inds: np.ndarray, dists: np.ndarray):
        """Build the merged graph"""
        n_samples = inds.shape[0]
        edge_weights = {}

        for i in range(n_samples):
            neigh_idx = inds[i]
            neigh_dist = dists[i]

            rho_i = float(neigh_dist[0]) if self.use_rho else 0.0
            sigma_i = max(float(neigh_dist[-1]), 1e-12)

            local_w = np.maximum(neigh_dist - rho_i, 0.0) / sigma_i

            for k, j in enumerate(neigh_idx):
                self.update_edge(edge_weights, i, int(j), float(local_w[k]))

            # UM connects dots not only with the center
            if self.mode == 'um':
                m = len(neigh_idx)
                for a_ind in range(m):
                    a = int(neigh_idx[a_ind])
                    wa = float(local_w[a_ind])
                    for b_ind in range(a_ind + 1, m):
                        b = int(neigh_idx[b_ind])
                        wb = float(local_w[b_ind])
                        self.update_edge(edge_weights, a, b, wa + wb)

        return edge_weights

    @staticmethod
    def edge_dict_to_csr(edge_weights: dict[tuple[int, int], float], n_samples: int):
        """Convert dict with edges to csr matrix"""
        # init diags
        rows = list(range(n_samples))
        cols = list(range(n_samples))
        data = [0.0] * n_samples

        # create other edges
        for (u, v), w in edge_weights.items():
            rows.extend([u, v])
            cols.extend([v, u])
            data.extend([w, w])

        return coo_matrix((data, (rows, cols)), shape=(n_samples, n_samples)).tocsr()

    @staticmethod
    def classical_mds(D: np.ndarray, n_components: int):
        """Classical MDS"""
        D = np.asarray(D, dtype=np.float64)
        n = D.shape[0]

        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError('D must be a square distance matrix with dim=2')
        if not np.allclose(D, D.T):
            raise ValueError('D must be symmetric')

        D2 = D**2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J

        eigvals, eigvecs = eigh(B)
        sorted_vals_ind = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_vals_ind]
        eigvecs = eigvecs[:, sorted_vals_ind]

        pos = (eigvals > 0)
        eigvals = eigvals[pos]
        eigvecs = eigvecs[:, pos]

        if eigvals.size == 0:
            raise ValueError('No positive eigenvalues found in classical MDS')

        r = min(n_components, eigvals.size)
        return eigvecs[:, :r] * np.sqrt(eigvals[:r])

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')

        # check data
        n_samples = X.shape[0]
        self._validate_params(n_samples)

        # find distances
        nn = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            metric=self.metric,
        )
        nn.fit(X)
        dists, inds = nn.kneighbors(X)
        dists = dists[:, 1:]
        inds = inds[:, 1:]

        # create merged graph
        edge_weights = self.build_merged_graph(inds, dists)
        graph = self.edge_dict_to_csr(edge_weights, n_samples)

        # update infty on the basis of shortest paths
        D = shortest_path(graph, directed=False, unweighted=False)
        if not np.isfinite(D).all():
            raise ValueError('Infinities in merged graph')

        # classic MDS
        Y0 = self.classical_mds(D, self.n_components)

        # metric MDS
        if self.refine_metric_mds:
            mds = MDS(
                n_components=self.n_components,
                metric=True,
                dissimilarity='precomputed',
                init=Y0,
                max_iter=self.mds_max_iter,
                n_init=self.mds_n_init,
                random_state=self.random_state,
                normalized_stress='auto',
            )
            Y = mds.fit_transform(D)
        else:
            Y = Y0
            self.stress_ = None

        self.nn = nn
        self.X_fit = X
        self.embedding = Y
        self.graph = graph
        self.geod_distances = D

        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).embedding
    