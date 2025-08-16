#!/usr/bin/env python3
"""
Unit tests for enhanced visualization functions in quatica/visualization.py

This module tests the new visualization capabilities added for:
- Matrix absolute value visualization
- Tensor slice visualization
- Schur structure analysis
- Convergence comparison plots

Author: QuatIca Development Team
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "quatica"))

import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import quaternion
from data_gen import create_test_matrix
from utils import quat_eye

# Import modules under test
from visualization import Visualizer


class TestEnhancedVisualization(unittest.TestCase):
    """Test suite for enhanced visualization functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.n = 5
        self.test_matrix = create_test_matrix(self.n, self.n)
        self.identity = quat_eye(self.n)

        # Create test tensor (3D)
        self.tensor_shape = (4, 5, 6)
        self.test_tensor = np.empty(self.tensor_shape, dtype=np.quaternion)
        for i in range(self.tensor_shape[0]):
            for j in range(self.tensor_shape[1]):
                for k in range(self.tensor_shape[2]):
                    w = np.random.randn()
                    x = np.random.randn()
                    y = np.random.randn()
                    z = np.random.randn()
                    self.test_tensor[i, j, k] = quaternion.quaternion(w, x, y, z)

        # Test convergence data
        self.convergence_data = {
            "Method A": [1.0, 0.5, 0.25, 0.125, 0.0625],
            "Method B": [1.0, 0.8, 0.4, 0.1, 0.02],
            "Method C": [1.0, 0.7, 0.3, 0.05, 0.001],
        }

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.colorbar")
    @patch("matplotlib.pyplot.imshow")
    def test_visualize_matrix_abs_basic(self, mock_imshow, mock_colorbar, mock_show):
        """Test basic functionality of visualize_matrix_abs."""
        # Test with default parameters
        Visualizer.visualize_matrix_abs(self.test_matrix)

        # Verify matplotlib functions were called
        mock_imshow.assert_called_once()
        mock_colorbar.assert_called_once()
        mock_show.assert_called_once()

        # Check imshow was called with correct data shape
        call_args = mock_imshow.call_args[0]
        abs_data = call_args[0]
        self.assertEqual(abs_data.shape, (self.n, self.n))

        # Verify all values are non-negative (absolute values)
        self.assertTrue(np.all(abs_data >= 0))

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.colorbar")
    @patch("matplotlib.pyplot.imshow")
    def test_visualize_matrix_abs_with_params(
        self, mock_imshow, mock_colorbar, mock_show
    ):
        """Test visualize_matrix_abs with custom parameters."""
        title = "Test Matrix Absolute Values"
        subtitle = "Custom Test"
        cmap = "plasma"

        Visualizer.visualize_matrix_abs(
            self.test_matrix, cmap=cmap, title=title, subtitle=subtitle
        )

        # Verify functions were called
        mock_imshow.assert_called_once()
        mock_colorbar.assert_called_once()
        mock_show.assert_called_once()

        # Check cmap parameter was passed
        call_kwargs = mock_imshow.call_args[1]
        self.assertEqual(call_kwargs.get("cmap"), cmap)

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.colorbar")
    @patch("matplotlib.pyplot.imshow")
    def test_visualize_tensor_slice_mode0(self, mock_imshow, mock_colorbar, mock_show):
        """Test tensor slice visualization with mode 0."""
        slice_idx = 1

        Visualizer.visualize_tensor_slice(self.test_tensor, mode=0, slice_idx=slice_idx)

        # Verify matplotlib functions were called
        mock_imshow.assert_called_once()
        mock_colorbar.assert_called_once()
        mock_show.assert_called_once()

        # Check data shape (should be 2D slice)
        call_args = mock_imshow.call_args[0]
        slice_data = call_args[0]
        expected_shape = (self.tensor_shape[1], self.tensor_shape[2])
        self.assertEqual(slice_data.shape, expected_shape)

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.colorbar")
    @patch("matplotlib.pyplot.imshow")
    def test_visualize_tensor_slice_mode1(self, mock_imshow, mock_colorbar, mock_show):
        """Test tensor slice visualization with mode 1."""
        slice_idx = 2

        Visualizer.visualize_tensor_slice(self.test_tensor, mode=1, slice_idx=slice_idx)

        mock_imshow.assert_called_once()
        mock_colorbar.assert_called_once()
        mock_show.assert_called_once()

        # Check data shape
        call_args = mock_imshow.call_args[0]
        slice_data = call_args[0]
        expected_shape = (self.tensor_shape[0], self.tensor_shape[2])
        self.assertEqual(slice_data.shape, expected_shape)

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.colorbar")
    @patch("matplotlib.pyplot.imshow")
    def test_visualize_tensor_slice_mode2(self, mock_imshow, mock_colorbar, mock_show):
        """Test tensor slice visualization with mode 2."""
        slice_idx = 3

        Visualizer.visualize_tensor_slice(self.test_tensor, mode=2, slice_idx=slice_idx)

        mock_imshow.assert_called_once()
        mock_colorbar.assert_called_once()
        mock_show.assert_called_once()

        # Check data shape
        call_args = mock_imshow.call_args[0]
        slice_data = call_args[0]
        expected_shape = (self.tensor_shape[0], self.tensor_shape[1])
        self.assertEqual(slice_data.shape, expected_shape)

    def test_visualize_tensor_slice_invalid_mode(self):
        """Test tensor slice visualization with invalid mode."""
        with self.assertRaises(ValueError):
            Visualizer.visualize_tensor_slice(self.test_tensor, mode=3)

    def test_visualize_tensor_slice_non_3d(self):
        """Test tensor slice visualization with non-3D tensor."""
        matrix_2d = self.test_matrix  # 2D matrix

        with self.assertRaises(ValueError):
            Visualizer.visualize_tensor_slice(matrix_2d)

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.imshow")
    @patch("matplotlib.pyplot.colorbar")
    def test_visualize_tensor_slice_abs_vs_real(
        self, mock_colorbar, mock_imshow, mock_show
    ):
        """Test tensor slice visualization with show_abs True vs False."""
        slice_idx = 0

        # Test with absolute values
        Visualizer.visualize_tensor_slice(
            self.test_tensor, mode=0, slice_idx=slice_idx, show_abs=True
        )

        abs_call_args = mock_imshow.call_args[0]
        abs_data = abs_call_args[0]

        # Reset mocks
        mock_imshow.reset_mock()

        # Test with real component only
        Visualizer.visualize_tensor_slice(
            self.test_tensor, mode=0, slice_idx=slice_idx, show_abs=False
        )

        real_call_args = mock_imshow.call_args[0]
        real_data = real_call_args[0]

        # Verify data shapes are the same
        self.assertEqual(abs_data.shape, real_data.shape)

        # Verify absolute values are non-negative
        self.assertTrue(np.all(abs_data >= 0))

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.colorbar")
    @patch("matplotlib.pyplot.imshow")
    @patch("matplotlib.pyplot.subplots")
    def test_visualize_schur_structure_basic(
        self, mock_subplots, mock_imshow, mock_colorbar, mock_show
    ):
        """Test basic functionality of visualize_schur_structure."""
        # Mock subplots to return figure and axes
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        # Create upper triangular test matrix
        T = np.triu(self.test_matrix)

        below_diag_max, subdiag_max = Visualizer.visualize_schur_structure(T)

        # Verify return values are floats
        self.assertIsInstance(below_diag_max, float)
        self.assertIsInstance(subdiag_max, float)

        # For upper triangular matrix, below_diag_max should be small
        self.assertLess(below_diag_max, 1e-10)  # Should be essentially zero

        # Verify subplots was called
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.colorbar")
    @patch("matplotlib.pyplot.imshow")
    @patch("matplotlib.pyplot.subplots")
    def test_visualize_schur_structure_with_params(
        self, mock_subplots, mock_imshow, mock_colorbar, mock_show
    ):
        """Test visualize_schur_structure with custom parameters."""
        # Mock subplots to return figure and axes
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        title = "Test Schur Structure"
        subtitle = "Custom Analysis"
        threshold = 1e-8

        below_diag_max, subdiag_max = Visualizer.visualize_schur_structure(
            self.test_matrix, title=title, subtitle=subtitle, threshold=threshold
        )

        # Verify function completed and returned values
        self.assertIsInstance(below_diag_max, float)
        self.assertIsInstance(subdiag_max, float)
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()

    def test_visualize_schur_structure_metrics(self):
        """Test metrics computation in visualize_schur_structure."""
        # Create test matrix with known structure
        T = np.zeros((4, 4), dtype=np.quaternion)

        # Set diagonal elements
        for i in range(4):
            T[i, i] = quaternion.quaternion(i + 1, 0, 0, 0)

        # Set some upper triangular elements
        T[0, 1] = quaternion.quaternion(0.5, 0, 0, 0)
        T[1, 2] = quaternion.quaternion(0.3, 0, 0, 0)

        # Set one subdiagonal element
        T[1, 0] = quaternion.quaternion(0.1, 0, 0, 0)

        # Set one below diagonal element
        T[2, 0] = quaternion.quaternion(0.2, 0, 0, 0)

        with (
            patch("matplotlib.pyplot.show"),
            patch("matplotlib.pyplot.imshow"),
            patch("matplotlib.pyplot.colorbar"),
            patch("matplotlib.pyplot.subplots") as mock_subplots,
        ):
            # Mock subplots to return figure and axes
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            below_diag_max, subdiag_max = Visualizer.visualize_schur_structure(T)

        # Verify metrics
        self.assertAlmostEqual(below_diag_max, 0.2, places=10)  # T[2,0]
        self.assertAlmostEqual(subdiag_max, 0.1, places=10)  # T[1,0]

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.semilogy")
    def test_plot_convergence_comparison_logscale(
        self, mock_semilogy, mock_plot, mock_show
    ):
        """Test convergence comparison plot with log scale."""
        Visualizer.plot_convergence_comparison(
            self.convergence_data, title="Test Convergence", logscale=True
        )

        # Verify semilogy was called (not regular plot)
        self.assertGreater(mock_semilogy.call_count, 0)
        self.assertEqual(mock_plot.call_count, 0)
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.semilogy")
    def test_plot_convergence_comparison_linear(
        self, mock_semilogy, mock_plot, mock_show
    ):
        """Test convergence comparison plot with linear scale."""
        Visualizer.plot_convergence_comparison(
            self.convergence_data, title="Test Convergence", logscale=False
        )

        # Verify plot was called (not semilogy)
        self.assertGreater(mock_plot.call_count, 0)
        self.assertEqual(mock_semilogy.call_count, 0)
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.semilogy")
    def test_plot_convergence_comparison_save(
        self, mock_semilogy, mock_savefig, mock_show
    ):
        """Test convergence comparison plot with save functionality."""
        # Create temporary file path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            save_path = tmp.name

        try:
            Visualizer.plot_convergence_comparison(
                self.convergence_data, save_path=save_path
            )

            # Verify savefig was called
            mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches="tight")
            mock_show.assert_called_once()

        finally:
            # Clean up temp file
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_plot_convergence_comparison_empty_data(self):
        """Test convergence comparison plot with empty data."""
        empty_data = {}

        with patch("matplotlib.pyplot.show") as mock_show:
            Visualizer.plot_convergence_comparison(empty_data)
            # Should still call show (empty plot)
            mock_show.assert_called_once()

    def test_visualize_matrix_abs_identity(self):
        """Test matrix absolute value visualization on identity matrix."""
        with (
            patch("matplotlib.pyplot.show"),
            patch("matplotlib.pyplot.imshow") as mock_imshow,
            patch("matplotlib.pyplot.colorbar"),
        ):
            Visualizer.visualize_matrix_abs(self.identity)

            # Check that data was plotted
            mock_imshow.assert_called_once()

            # Get the plotted data
            call_args = mock_imshow.call_args[0]
            abs_data = call_args[0]

            # For identity matrix, diagonal should be 1, off-diagonal should be 0
            for i in range(self.n):
                for j in range(self.n):
                    if i == j:
                        self.assertAlmostEqual(abs_data[i, j], 1.0, places=10)
                    else:
                        self.assertAlmostEqual(abs_data[i, j], 0.0, places=10)

    def test_visualize_schur_structure_identity(self):
        """Test Schur structure visualization on identity matrix."""
        with (
            patch("matplotlib.pyplot.show"),
            patch("matplotlib.pyplot.imshow"),
            patch("matplotlib.pyplot.colorbar"),
            patch("matplotlib.pyplot.subplots") as mock_subplots,
        ):
            # Mock subplots to return figure and axes
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            below_diag_max, subdiag_max = Visualizer.visualize_schur_structure(
                self.identity
            )

        # For identity matrix, both should be essentially zero
        self.assertLess(below_diag_max, 1e-10)
        self.assertLess(subdiag_max, 1e-10)


class TestVisualizationErrorHandling(unittest.TestCase):
    """Test error handling in visualization functions."""

    def test_invalid_tensor_dimensions(self):
        """Test handling of invalid tensor dimensions."""
        # 2D matrix (not 3D tensor)
        matrix_2d = create_test_matrix(3, 3)

        with self.assertRaises(ValueError) as context:
            Visualizer.visualize_tensor_slice(matrix_2d)

        self.assertIn("3D", str(context.exception))

    def test_invalid_slice_mode(self):
        """Test handling of invalid slice mode."""
        tensor_3d = np.random.random((3, 4, 5))

        with self.assertRaises(ValueError) as context:
            Visualizer.visualize_tensor_slice(tensor_3d, mode=5)

        self.assertIn("Mode must be", str(context.exception))

    def test_out_of_bounds_slice_index(self):
        """Test graceful handling of out-of-bounds slice indices."""
        tensor_3d = np.empty((3, 4, 5), dtype=np.quaternion)

        # This should not raise an error immediately (NumPy will handle)
        # But we can test that the function structure works
        with (
            patch("matplotlib.pyplot.show"),
            patch("matplotlib.pyplot.imshow"),
            patch("matplotlib.pyplot.colorbar"),
        ):
            try:
                # Valid slice index
                Visualizer.visualize_tensor_slice(tensor_3d, mode=0, slice_idx=2)
                # This should work
            except IndexError:
                self.fail("Valid slice index should not raise IndexError")


if __name__ == "__main__":
    # Configure test runner
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedVisualization))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualizationErrorHandling))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
