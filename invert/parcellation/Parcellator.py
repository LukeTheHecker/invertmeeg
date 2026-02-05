import copy
import logging
from pathlib import Path
from typing import Optional

import mne
import numpy as np
from scipy.sparse import coo_matrix

logger = logging.getLogger(__name__)


class Parcellator:
    def __init__(
        self,
        forward: mne.Forward,
        adjacency: coo_matrix,
        subjects_dir: Optional[str] = None,
    ):
        """
        Initialize the Parcellator.

        Parameters
        ----------
        forward : mne.Forward
            The forward solution.
        adjacency : coo_matrix
            The adjacency matrix.
        subjects_dir : str, optional
            Path to the subjects directory. If None, uses MNE's default.
        """
        self.forward = forward
        self.adjacency = adjacency
        self.leadfield = forward["sol"]["data"]
        self.leadfield_whitened = self.whiten(self.leadfield)
        self.hemispheres = ["lh", "rh"]
        self.B = None  # Will be set after parcellation
        self.forward_parcellated = None
        self.parcel_adjacency = None  # Will be set after parcellation

        if subjects_dir is None:
            self.subjects_dir = Path(mne.datasets.sample.data_path()) / "subjects"
        else:
            self.subjects_dir = subjects_dir  # type: ignore[assignment]

        self.src = self.forward["src"]
        self.vertices = [self.src[0]["vertno"], self.src[1]["vertno"]]

    def parcellate(
        self, parcellation: str = "aparc.a2009s", subject: Optional[str] = None
    ):
        """
        Parcellate the leadfield using the given parcellation scheme.

        Parameters
        ----------
        parcellation : str
            The parcellation to use (default: "aparc.a2009s").
        subject : str, optional
            Subject name. If None, extracts from forward solution.

        Returns
        -------
        fwd_parcellated : mne.Forward
            Parcellated forward solution.

        """
        # Extract subject from forward solution if not provided
        if subject is None:
            subject = self.forward["src"][0].get("subject_his_id", "fsaverage")

        # Read labels for both hemispheres and flatten the list
        labels = []
        for hemisphere in self.hemispheres:
            #     hemi_labels = mne.read_labels_from_annot(
            #         subject,
            #         parc=parcellation,
            #         hemi=hemisphere,
            #         subjects_dir=self.subjects_dir
            #     )
            hemi_labels = mne.read_labels_from_annot(
                subject, parcellation, hemisphere, subjects_dir=self.subjects_dir
            )
            labels.extend(hemi_labels)

        # Build parcellation basis and get representative vertices for each parcel
        self.B, parcel_vertices_lh, parcel_vertices_rh = (
            self._parcellation_basis_from_labels(labels)
        )

        # Compute parcellated leadfield
        L_B = self.leadfield @ self.B

        # Compute parcel-to-parcel adjacency
        self.parcel_adjacency = self._compute_parcel_adjacency()

        # Create a copy of the forward solution and modify its leadfield
        fwd_parcellated = copy.deepcopy(self.forward)
        fwd_parcellated["sol"]["data"] = L_B
        if self.B is not None:
            fwd_parcellated["nsource"] = self.B.shape[1]
        fwd_parcellated["source_nn"] = np.eye(3)[
            :, [2]
        ]  # Simplified, as parcels don't have orientations

        # Update the source space vertices to match the parcellated space
        fwd_parcellated["src"][0]["vertno"] = parcel_vertices_lh
        fwd_parcellated["src"][0]["nuse"] = len(parcel_vertices_lh)
        if len(fwd_parcellated["src"]) > 1:
            fwd_parcellated["src"][1]["vertno"] = parcel_vertices_rh
            fwd_parcellated["src"][1]["nuse"] = len(parcel_vertices_rh)

        # Store for later use
        self.forward_parcellated = fwd_parcellated
        self.parcel_vertices = [parcel_vertices_lh, parcel_vertices_rh]

        return fwd_parcellated

    def compress(self, stc: mne.SourceEstimate) -> mne.SourceEstimate:
        """
        Compress the source estimate using the parcel basis.

        Parameters
        ----------
        stc : mne.SourceEstimate
            The source estimate (n_dipoles x n_timepoints).

        Returns
        -------
        stc_parcellated : mne.SourceEstimate
            The compressed source estimate (n_parcels x n_timepoints).
        """
        if self.B is None:
            raise ValueError("Must call parcellate() before compress()")
        if not hasattr(self, "parcel_vertices"):
            raise ValueError(
                "parcel_vertices not found. Make sure parcellate() completed successfully."
            )

        X = stc.data
        X_parcellated = self.compress_data(X)
        stc_parcellated = mne.SourceEstimate(
            X_parcellated,
            vertices=self.parcel_vertices,
            tmin=stc.tmin,
            tstep=stc.tstep,
            subject=stc.subject,
        )
        return stc_parcellated

    def compress_data(self, data: np.ndarray) -> np.ndarray:
        """
        Compress the data using the parcel basis.
        """
        assert self.B is not None
        return self.B.T @ data

    def decompress(self, stc_parcellated: mne.SourceEstimate) -> mne.SourceEstimate:
        """
        Decompress the source estimate from the parcel basis.

        Parameters
        ----------
        stc_parcellated : mne.SourceEstimate
            The compressed parcel data (n_parcels x n_timepoints).

        Returns
        -------
        stc : mne.SourceEstimate
            The decompressed source estimate (n_dipoles x n_timepoints).
        """
        if self.B is None:
            raise ValueError("Must call parcellate() before decompress()")

        X_parcellated = stc_parcellated.data
        X = self.decompress_data(
            X_parcellated
        )  # Project from parcel space back to dipole space
        stc = mne.SourceEstimate(
            X,
            vertices=self.vertices,
            tmin=stc_parcellated.tmin,
            tstep=stc_parcellated.tstep,
            subject=stc_parcellated.subject,
        )
        return stc

    def decompress_data(self, data: np.ndarray) -> np.ndarray:
        """
        Decompress the data using the parcel basis.
        """
        return self.B @ data

    def get_parcel_adjacency(self) -> coo_matrix:
        """
        Get the parcel-to-parcel adjacency matrix.

        Returns
        -------
        parcel_adjacency : coo_matrix
            Sparse adjacency matrix for parcels (n_parcels x n_parcels).

        Raises
        ------
        ValueError
            If parcellate() has not been called yet.
        """
        if self.parcel_adjacency is None:
            raise ValueError("Must call parcellate() before getting parcel adjacency")
        return self.parcel_adjacency

    def _parcellation_basis_from_labels(self, labels: list) -> tuple:
        """
        Build a parcel basis matrix B (n_dipoles x n_labels)
        mapping dipoles in the forward model to parcellation regions.

        Each column corresponds to one label (ROI), normalized to unit L2 norm.
        Also returns representative vertices for each parcel.

        Parameters
        ----------
        labels : list of mne.Label
            List of labels defining the parcellation.

        Returns
        -------
        B : np.ndarray
            Parcellation basis matrix (n_dipoles x n_labels).
        parcel_vertices_lh : np.ndarray
            Representative vertices for left hemisphere parcels (sorted).
        parcel_vertices_rh : np.ndarray
            Representative vertices for right hemisphere parcels (sorted).
        """
        src = self.forward["src"]
        lh = src[0]["vertno"]
        rh = src[1]["vertno"] if len(src) > 1 else np.array([], int)
        n_dip = len(lh) + len(rh)

        # Create a lookup: dipole index in full source space -> position in B
        offset_rh = len(lh)

        # First pass: collect parcel info for each hemisphere
        lh_parcels = []  # List of (representative_vertex, label, label_idx)
        rh_parcels = []

        for k, label in enumerate(labels):
            if label.hemi == "lh":
                verts = np.intersect1d(label.vertices, lh)
                if len(verts) > 0:
                    lh_parcels.append((verts[0], label, k, verts))
            elif label.hemi == "rh":
                verts = np.intersect1d(label.vertices, rh)
                if len(verts) > 0:
                    rh_parcels.append((verts[0], label, k, verts))

        # Sort by representative vertex to ensure vertices are in increasing order
        lh_parcels.sort(key=lambda x: x[0])
        rh_parcels.sort(key=lambda x: x[0])

        # Build B matrix in sorted order
        n_parcels = len(lh_parcels) + len(rh_parcels)
        B = np.zeros((n_dip, n_parcels))

        parcel_verts_lh = []
        parcel_verts_rh = []

        # Fill in left hemisphere parcels
        parcel_idx = 0
        for rep_vert, _label, _orig_idx, verts in lh_parcels:
            parcel_verts_lh.append(rep_vert)
            idx = np.searchsorted(lh, verts)
            B[idx, parcel_idx] = 1.0 / np.sqrt(len(idx))
            parcel_idx += 1

        # Fill in right hemisphere parcels
        for rep_vert, _label, _orig_idx, verts in rh_parcels:
            parcel_verts_rh.append(rep_vert)
            idx = np.searchsorted(rh, verts) + offset_rh
            B[idx, parcel_idx] = 1.0 / np.sqrt(len(idx))
            parcel_idx += 1

        return (
            B,
            np.array(parcel_verts_lh, dtype=int),
            np.array(parcel_verts_rh, dtype=int),
        )

    def _compute_parcel_adjacency(self) -> coo_matrix:
        """
        Compute parcel-to-parcel adjacency based on the original source space adjacency.

        Two parcels are considered adjacent if any of their constituent vertices
        are adjacent in the original source space.

        Returns
        -------
        parcel_adj : coo_matrix
            Sparse adjacency matrix for parcels (n_parcels x n_parcels).
        """
        if self.B is None:
            raise ValueError("Must build parcellation basis first")

        n_parcels = self.B.shape[1]
        n_vertices = self.B.shape[0]

        # Build vertex-to-parcel mapping: which parcel does each vertex belong to?
        vertex_to_parcel = np.full(n_vertices, -1, dtype=int)
        for vertex_idx in range(n_vertices):
            parcel_indices = np.nonzero(self.B[vertex_idx, :])[0]
            if len(parcel_indices) > 0:
                vertex_to_parcel[vertex_idx] = parcel_indices[0]

        # Make adjacency symmetric to capture all edges (handles both directions)
        adj_csr = self.adjacency.tocsr()
        adj_symmetric = adj_csr + adj_csr.T
        adj_coo = adj_symmetric.tocoo()

        # Collect parcel edges
        parcel_edges = set()
        for i, j in zip(adj_coo.row, adj_coo.col):
            # Skip if either vertex is not in any parcel (shouldn't happen, but be safe)
            if vertex_to_parcel[i] < 0 or vertex_to_parcel[j] < 0:
                continue

            parcel_i = vertex_to_parcel[i]
            parcel_j = vertex_to_parcel[j]

            # Only add if different parcels
            if parcel_i != parcel_j:
                parcel_edges.add((parcel_i, parcel_j))
                parcel_edges.add((parcel_j, parcel_i))

        # Convert to COO matrix
        if len(parcel_edges) > 0:
            edges = np.array(list(parcel_edges))
            row = edges[:, 0]
            col = edges[:, 1]
            data = np.ones(len(row))
            parcel_adj = coo_matrix((data, (row, col)), shape=(n_parcels, n_parcels))
        else:
            parcel_adj = coo_matrix((n_parcels, n_parcels))

        # Debug: Check if any parcels have no neighbors
        parcel_degrees = np.array(parcel_adj.sum(axis=1)).flatten()
        isolated_parcels = np.where(parcel_degrees == 0)[0]
        if len(isolated_parcels) > 0:
            logger.warning(
                f"{len(isolated_parcels)} parcel(s) have no neighbors: {isolated_parcels}"
            )
            # This could indicate vertices with no edges in the adjacency matrix
            # or parcels at the mesh boundary

        return parcel_adj

    def diagnose_parcel_connectivity(self) -> dict:
        """
        Diagnose connectivity issues in parcel adjacency.

        Returns
        -------
        diagnostics : dict
            Contains:
            - 'n_parcels': number of parcels
            - 'n_with_neighbors': number of parcels with at least one neighbor
            - 'n_isolated': number of parcels with no neighbors
            - 'isolated_parcel_ids': list of isolated parcel indices
            - 'parcel_degrees': array of neighbor counts per parcel
            - 'isolated_vertices_per_parcel': dict mapping parcel id to list of isolated vertex indices
        """
        if self.B is None:
            raise ValueError(
                "Must call parcellate() before diagnose_parcel_connectivity()"
            )

        n_parcels = self.B.shape[1]
        n_vertices = self.B.shape[0]

        # Get adjacency info
        adj_csr = self.adjacency.tocsr()
        adj_symmetric = adj_csr + adj_csr.T

        # Vertex-to-parcel mapping
        vertex_to_parcel = np.full(n_vertices, -1, dtype=int)
        for vertex_idx in range(n_vertices):
            parcel_indices = np.nonzero(self.B[vertex_idx, :])[0]
            if len(parcel_indices) > 0:
                vertex_to_parcel[vertex_idx] = parcel_indices[0]

        # Count neighbors for each vertex
        vertex_degrees = np.array(adj_symmetric.sum(axis=1)).flatten()

        # Find isolated vertices per parcel
        isolated_verts_per_parcel = {}
        for parcel_id in range(n_parcels):
            # Find all vertices in this parcel
            vertex_indices = np.where(vertex_to_parcel == parcel_id)[0]
            # Find which of these have no neighbors (degree == 0)
            isolated = vertex_indices[vertex_degrees[vertex_indices] == 0]
            if len(isolated) > 0:
                isolated_verts_per_parcel[parcel_id] = isolated.tolist()

        # Get parcel degrees from adjacency
        parcel_adj = (
            self.get_parcel_adjacency()
            if self.parcel_adjacency is not None
            else self._compute_parcel_adjacency()
        )
        parcel_degrees = np.array(parcel_adj.sum(axis=1)).flatten()

        isolated_parcels = np.where(parcel_degrees == 0)[0]

        return {
            "n_parcels": n_parcels,
            "n_with_neighbors": len(parcel_degrees) - len(isolated_parcels),
            "n_isolated": len(isolated_parcels),
            "isolated_parcel_ids": isolated_parcels.tolist(),
            "parcel_degrees": parcel_degrees,
            "isolated_vertices_per_parcel": isolated_verts_per_parcel,
        }

    @staticmethod
    def whiten(leadfield: np.ndarray) -> np.ndarray:
        """
        Whiten the leadfield by normalizing each column.

        Parameters
        ----------
        leadfield : np.ndarray
            The leadfield matrix (n_sensors x n_dipoles).

        Returns
        -------
        leadfield_whitened : np.ndarray
            The whitened leadfield matrix.
        """
        return leadfield / np.linalg.norm(leadfield, axis=0, keepdims=True)

    @staticmethod
    def trim_to_roi(
        fwd: mne.Forward,
        parcellation: str = "HCPMMP1_combined",
        roi: str = "front",
        subject: str = "fsaverage",
        subjects_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> mne.Forward:
        """
        Trim a forward solution to only include vertices in a specific ROI.

        This is useful for computing beamformers or other inverse solutions
        only in regions of interest.

        Parameters
        ----------
        fwd : mne.Forward
            The forward solution to trim.
        parcellation : str, optional
            The parcellation to use (default: "HCPMMP1_combined").
        roi : str, optional
            ROI specification. Labels containing this string (case-insensitive)
            will be included. Examples: 'front' for frontal lobe, 'temporal'
            for temporal lobe, 'precentral' for motor cortex (default: "front").
        subject : str, optional
            Subject name (default: "fsaverage").
        subjects_dir : str, optional
            Path to the subjects directory. If None, uses MNE's default.
        verbose : bool, optional
            Whether to print information about the trimming (default: True).

        Returns
        -------
        fwd_roi : mne.Forward
            Trimmed forward solution containing only vertices in the ROI.

        Raises
        ------
        ValueError
            If no labels matching the ROI are found.
            If no vertices in the forward solution belong to the ROI.
        """
        # Load parcellation labels
        labels_lh = mne.read_labels_from_annot(
            subject,
            parc=parcellation,
            hemi="lh",
            subjects_dir=subjects_dir,
            verbose=False,
        )
        labels_rh = mne.read_labels_from_annot(
            subject,
            parc=parcellation,
            hemi="rh",
            subjects_dir=subjects_dir,
            verbose=False,
        )
        all_labels = labels_lh + labels_rh

        # Find labels matching the ROI specification
        roi_labels = [
            label for label in all_labels if roi.lower() in label.name.lower()
        ]

        return mne.forward.restrict_forward_to_label(fwd, roi_labels)
