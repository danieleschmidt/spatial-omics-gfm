"""
Enhanced Basic Example - Pure Python Implementation

Demonstrates core functionality without external dependencies.
This serves as Generation 1 (Simple) implementation enhancement.
"""

import math
import json
import random
from typing import Dict, List, Any, Optional, Tuple


class PurePythonSpatialData:
    """Pure Python spatial transcriptomics data representation."""
    
    def __init__(
        self,
        expression_matrix: List[List[float]],
        coordinates: List[List[float]],
        gene_names: Optional[List[str]] = None
    ):
        """
        Initialize spatial data without external dependencies.
        
        Args:
            expression_matrix: Cell x Gene expression matrix (list of lists)
            coordinates: Cell x 2 coordinate matrix (list of [x, y] pairs)
            gene_names: Optional list of gene names
        """
        self.expression_matrix = expression_matrix
        self.coordinates = coordinates
        self.n_cells = len(expression_matrix)
        self.n_genes = len(expression_matrix[0]) if expression_matrix else 0
        self.gene_names = gene_names or [f"Gene_{i}" for i in range(self.n_genes)]
        
        # Validate data
        assert len(coordinates) == self.n_cells, "Coordinate count must match cell count"
        assert all(len(coord) == 2 for coord in coordinates), "Coordinates must be 2D (x, y)"
        assert all(len(row) == self.n_genes for row in expression_matrix), "All expression rows must have same length"
    
    def normalize_expression(self) -> None:
        """Normalize expression data using log transformation."""
        for i in range(self.n_cells):
            # Calculate cell total
            cell_total = sum(self.expression_matrix[i])
            if cell_total == 0:
                cell_total = 1  # Avoid division by zero
            
            # Normalize to counts per 10,000 and log transform
            for j in range(self.n_genes):
                normalized_value = (self.expression_matrix[i][j] / cell_total) * 10000
                self.expression_matrix[i][j] = math.log1p(normalized_value)
    
    def calculate_distance(self, coord1: List[float], coord2: List[float]) -> float:
        """Calculate Euclidean distance between two coordinates."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
    
    def find_spatial_neighbors(self, k: int = 6) -> Dict[int, List[int]]:
        """Find k nearest spatial neighbors for each cell."""
        neighbors = {}
        
        for i in range(self.n_cells):
            # Calculate distances to all other cells
            distances = []
            for j in range(self.n_cells):
                if i != j:
                    dist = self.calculate_distance(self.coordinates[i], self.coordinates[j])
                    distances.append((dist, j))
            
            # Sort by distance and take k nearest
            distances.sort(key=lambda x: x[0])
            neighbors[i] = [idx for _, idx in distances[:k]]
        
        return neighbors
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get basic summary statistics."""
        # Calculate means
        cell_totals = [sum(row) for row in self.expression_matrix]
        gene_totals = [sum(self.expression_matrix[i][j] for i in range(self.n_cells)) 
                      for j in range(self.n_genes)]
        
        # Spatial extents
        x_coords = [coord[0] for coord in self.coordinates]
        y_coords = [coord[1] for coord in self.coordinates]
        
        return {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "mean_expression_per_cell": sum(cell_totals) / len(cell_totals) if cell_totals else 0,
            "mean_expression_per_gene": sum(gene_totals) / len(gene_totals) if gene_totals else 0,
            "spatial_extent": {
                "x_range": [min(x_coords), max(x_coords)] if x_coords else [0, 0],
                "y_range": [min(y_coords), max(y_coords)] if y_coords else [0, 0]
            }
        }


class EnhancedCellTypePredictor:
    """Enhanced cell type prediction with more sophisticated scoring."""
    
    def __init__(self):
        self.marker_genes = {
            "T_cell": ["CD3D", "CD3E", "CD8A", "CD4", "TRAC", "TRBC1"],
            "B_cell": ["CD19", "CD20", "MS4A1", "CD79A", "CD79B", "IGHM"],
            "Macrophage": ["CD68", "CD163", "CSF1R", "ADGRE1", "AIF1", "LYZ"],
            "Fibroblast": ["COL1A1", "COL3A1", "FN1", "VIM", "DCN", "LUM"],
            "Epithelial": ["EPCAM", "KRT18", "KRT19", "CDH1", "KRT8", "KRT7"],
            "Endothelial": ["PECAM1", "VWF", "ENG", "PLVAP", "FLT1", "KDR"],
            "NK_cell": ["KLRB1", "NCR1", "GNLY", "NKG7", "PRF1", "GZMB"],
            "DC_cell": ["CD1C", "CLEC9A", "IRF8", "BATF3", "FCER1A", "CD1A"]
        }
        
        # Cell type specificity weights (higher = more specific)
        self.specificity_weights = {
            "T_cell": 0.9,
            "B_cell": 0.95,
            "Macrophage": 0.8,
            "Fibroblast": 0.7,
            "Epithelial": 0.85,
            "Endothelial": 0.9,
            "NK_cell": 0.95,
            "DC_cell": 0.9
        }
    
    def predict_cell_types(self, data: PurePythonSpatialData) -> Dict[str, List[float]]:
        """
        Enhanced cell type prediction with weighted scoring.
        
        Returns:
            Dictionary mapping cell type names to confidence scores per cell
        """
        predictions = {}
        
        for cell_type, markers in self.marker_genes.items():
            # Find available marker genes
            available_markers = [gene for gene in markers if gene in data.gene_names]
            
            if available_markers:
                # Get marker indices
                marker_indices = [data.gene_names.index(gene) for gene in available_markers]
                
                # Calculate weighted expression scores
                cell_scores = []
                for i in range(data.n_cells):
                    # Get marker expressions for this cell
                    marker_expressions = [data.expression_matrix[i][idx] for idx in marker_indices]
                    
                    # Calculate geometric mean for robustness
                    if marker_expressions:
                        # Add small epsilon to avoid log(0)
                        log_expressions = [math.log(expr + 0.01) for expr in marker_expressions]
                        geo_mean = math.exp(sum(log_expressions) / len(log_expressions))
                        
                        # Apply specificity weight
                        weighted_score = geo_mean * self.specificity_weights.get(cell_type, 1.0)
                        cell_scores.append(weighted_score)
                    else:
                        cell_scores.append(0.01)
                
                # Normalize scores to [0, 1] range
                max_score = max(cell_scores) if cell_scores else 1.0
                if max_score > 0:
                    cell_scores = [score / max_score for score in cell_scores]
                
                predictions[cell_type] = cell_scores
            else:
                # No markers available - assign low random scores
                predictions[cell_type] = [random.random() * 0.1 for _ in range(data.n_cells)]
        
        return predictions
    
    def assign_best_cell_types(self, predictions: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Assign the most likely cell type to each cell."""
        n_cells = len(next(iter(predictions.values())))
        assignments = []
        
        for i in range(n_cells):
            # Get scores for this cell across all types
            cell_scores = {cell_type: scores[i] for cell_type, scores in predictions.items()}
            
            # Find best match
            best_type = max(cell_scores.keys(), key=lambda x: cell_scores[x])
            best_score = cell_scores[best_type]
            
            # Calculate confidence (difference from second best)
            sorted_scores = sorted(cell_scores.values(), reverse=True)
            confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
            
            assignments.append({
                "cell_id": i,
                "predicted_type": best_type,
                "confidence": confidence,
                "all_scores": cell_scores.copy()
            })
        
        return assignments


class EnhancedInteractionPredictor:
    """Enhanced cell-cell interaction prediction with pathway context."""
    
    def __init__(self):
        # Expanded ligand-receptor database
        self.ligand_receptor_pairs = [
            # Immune signaling
            ("CXCL12", "CXCR4"), ("CCL2", "CCR2"), ("CCL5", "CCR5"),
            ("TNF", "TNFRSF1A"), ("IL1B", "IL1R1"), ("IL6", "IL6R"),
            ("IFNG", "IFNGR1"), ("IL10", "IL10RA"), ("IL4", "IL4R"),
            
            # Growth factors
            ("VEGFA", "KDR"), ("VEGFA", "FLT1"), ("FGF2", "FGFR1"),
            ("PDGFA", "PDGFRA"), ("EGF", "EGFR"), ("IGF1", "IGF1R"),
            
            # ECM and adhesion
            ("TGFB1", "TGFBR1"), ("BMP2", "BMPR1A"), ("WNT3A", "FZD1"),
            ("DLL4", "NOTCH1"), ("JAG1", "NOTCH2"), ("SLIT2", "ROBO1"),
            
            # Metabolic
            ("LEP", "LEPR"), ("INS", "INSR"), ("GCG", "GCGR")
        ]
        
        # Interaction pathway categories
        self.pathway_categories = {
            "immune_activation": [("TNF", "TNFRSF1A"), ("IL1B", "IL1R1"), ("IFNG", "IFNGR1")],
            "immune_suppression": [("IL10", "IL10RA"), ("TGFB1", "TGFBR1")],
            "chemotaxis": [("CXCL12", "CXCR4"), ("CCL2", "CCR2"), ("CCL5", "CCR5")],
            "angiogenesis": [("VEGFA", "KDR"), ("VEGFA", "FLT1"), ("FGF2", "FGFR1")],
            "growth_signaling": [("EGF", "EGFR"), ("IGF1", "IGF1R"), ("PDGFA", "PDGFRA")],
            "developmental": [("WNT3A", "FZD1"), ("DLL4", "NOTCH1"), ("BMP2", "BMPR1A")]
        }
    
    def calculate_interaction_score(
        self, 
        ligand_expr: float, 
        receptor_expr: float, 
        distance: float,
        max_distance: float = 100.0
    ) -> float:
        """Calculate interaction score with distance decay."""
        # Base interaction strength
        base_score = ligand_expr * receptor_expr
        
        # Distance decay (exponential)
        distance_factor = math.exp(-distance / (max_distance / 3))
        
        return base_score * distance_factor
    
    def predict_interactions(
        self, 
        data: PurePythonSpatialData, 
        max_distance: float = 100.0,
        min_score: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Enhanced interaction prediction with pathway categorization.
        
        Args:
            data: Spatial data
            max_distance: Maximum distance for interaction consideration
            min_score: Minimum interaction score threshold
            
        Returns:
            List of predicted interactions with metadata
        """
        interactions = []
        neighbors = data.find_spatial_neighbors(k=15)  # Increased neighborhood size
        
        for cell_i in range(data.n_cells):
            for neighbor_j in neighbors.get(cell_i, []):
                # Calculate distance
                distance = data.calculate_distance(
                    data.coordinates[cell_i], 
                    data.coordinates[neighbor_j]
                )
                
                if distance <= max_distance:
                    # Check all ligand-receptor pairs
                    for ligand, receptor in self.ligand_receptor_pairs:
                        if ligand in data.gene_names and receptor in data.gene_names:
                            ligand_idx = data.gene_names.index(ligand)
                            receptor_idx = data.gene_names.index(receptor)
                            
                            ligand_expr = data.expression_matrix[cell_i][ligand_idx]
                            receptor_expr = data.expression_matrix[neighbor_j][receptor_idx]
                            
                            # Calculate enhanced interaction score
                            interaction_score = self.calculate_interaction_score(
                                ligand_expr, receptor_expr, distance, max_distance
                            )
                            
                            if interaction_score >= min_score:
                                # Find pathway category
                                pathway = "other"
                                for cat, pairs in self.pathway_categories.items():
                                    if (ligand, receptor) in pairs:
                                        pathway = cat
                                        break
                                
                                interactions.append({
                                    "sender_cell": int(cell_i),
                                    "receiver_cell": int(neighbor_j),
                                    "ligand": ligand,
                                    "receptor": receptor,
                                    "interaction_score": float(interaction_score),
                                    "distance": float(distance),
                                    "pathway_category": pathway,
                                    "ligand_expression": float(ligand_expr),
                                    "receptor_expression": float(receptor_expr)
                                })
        
        return interactions
    
    def analyze_pathway_enrichment(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze pathway enrichment in interactions."""
        pathway_counts = {}
        total_interactions = len(interactions)
        
        for interaction in interactions:
            pathway = interaction.get("pathway_category", "other")
            pathway_counts[pathway] = pathway_counts.get(pathway, 0) + 1
        
        # Calculate enrichment statistics
        enrichment_stats = {}
        for pathway, count in pathway_counts.items():
            enrichment_stats[pathway] = {
                "count": count,
                "frequency": count / total_interactions if total_interactions > 0 else 0,
                "mean_score": sum(
                    interaction["interaction_score"] 
                    for interaction in interactions 
                    if interaction.get("pathway_category") == pathway
                ) / count if count > 0 else 0
            }
        
        return enrichment_stats


def create_enhanced_demo_data(n_cells: int = 1000, n_genes: int = 500) -> PurePythonSpatialData:
    """Create enhanced synthetic spatial transcriptomics data."""
    
    # Generate structured coordinates (tissue-like patterns)
    coordinates = []
    
    # Create multiple tissue regions
    regions = [
        {"center": [200, 200], "radius": 150, "n_cells": n_cells // 3},  # Central region
        {"center": [600, 200], "radius": 100, "n_cells": n_cells // 4},  # Right region  
        {"center": [400, 600], "radius": 120, "n_cells": n_cells // 4},  # Bottom region
    ]
    
    cell_region_assignments = []
    
    # Generate coordinates for each region
    for region_idx, region in enumerate(regions):
        for _ in range(region["n_cells"]):
            # Generate coordinates within circular region
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, region["radius"]) * math.sqrt(random.random())
            
            x = region["center"][0] + radius * math.cos(angle)
            y = region["center"][1] + radius * math.sin(angle)
            
            coordinates.append([x, y])
            cell_region_assignments.append(region_idx)
    
    # Fill remaining cells randomly
    remaining_cells = n_cells - len(coordinates)
    for _ in range(remaining_cells):
        x = random.uniform(0, 800)
        y = random.uniform(0, 800)
        coordinates.append([x, y])
        cell_region_assignments.append(-1)  # Unassigned
    
    # Create realistic gene names
    important_genes = [
        # T cell markers
        "CD3D", "CD3E", "CD8A", "CD4", "TRAC", "TRBC1",
        # B cell markers  
        "CD19", "CD20", "MS4A1", "CD79A", "CD79B", "IGHM",
        # Macrophage markers
        "CD68", "CD163", "CSF1R", "ADGRE1", "AIF1", "LYZ",
        # Fibroblast markers
        "COL1A1", "COL3A1", "FN1", "VIM", "DCN", "LUM",
        # Epithelial markers
        "EPCAM", "KRT18", "KRT19", "CDH1", "KRT8", "KRT7",
        # Endothelial markers
        "PECAM1", "VWF", "ENG", "PLVAP", "FLT1", "KDR",
        # Interaction genes
        "CXCL12", "CXCR4", "CCL2", "CCR2", "TNF", "TNFRSF1A",
        "IL1B", "IL1R1", "VEGFA", "TGFB1", "TGFBR1", "IFNG", "IFNGR1"
    ]
    
    other_genes = [f"Gene_{i:04d}" for i in range(n_genes - len(important_genes))]
    gene_names = important_genes + other_genes
    gene_names = gene_names[:n_genes]  # Ensure correct length
    
    # Generate expression matrix with regional patterns
    expression_matrix = []
    
    for i in range(n_cells):
        cell_expression = []
        region_idx = cell_region_assignments[i] if i < len(cell_region_assignments) else -1
        
        for j, gene in enumerate(gene_names):
            # Base expression
            base_expr = max(0, random.gauss(2.0, 1.5))  # Log-normal like distribution
            
            # Add regional specificity for marker genes
            if gene in important_genes:
                if region_idx == 0 and gene in ["CD3D", "CD3E", "CD8A", "CD4"]:
                    # T cells enriched in region 0
                    base_expr += random.uniform(2, 5)
                elif region_idx == 1 and gene in ["CD68", "CD163", "CSF1R"]:
                    # Macrophages enriched in region 1
                    base_expr += random.uniform(2, 4)
                elif region_idx == 2 and gene in ["COL1A1", "COL3A1", "FN1"]:
                    # Fibroblasts enriched in region 2
                    base_expr += random.uniform(1.5, 3)
            
            cell_expression.append(max(0, base_expr))
        
        expression_matrix.append(cell_expression)
    
    return PurePythonSpatialData(expression_matrix, coordinates, gene_names)


def run_enhanced_analysis(data: Optional[PurePythonSpatialData] = None) -> Dict[str, Any]:
    """Run enhanced comprehensive analysis - Generation 1 implementation."""
    print("=== Enhanced Spatial-Omics GFM Analysis (Generation 1) ===")
    
    # Create or use provided demo data
    if data is None:
        print("1. Creating enhanced synthetic spatial transcriptomics data...")
        data = create_enhanced_demo_data(n_cells=800, n_genes=150)
    else:
        print("1. Using provided spatial transcriptomics data...")
    
    # Normalize data
    print("2. Normalizing expression data with log transformation...")
    data.normalize_expression()
    
    # Get comprehensive summary stats
    print("3. Computing comprehensive summary statistics...")
    stats = data.get_summary_stats()
    print(f"   - Cells: {stats['n_cells']}")
    print(f"   - Genes: {stats['n_genes']}")
    print(f"   - Mean expression per cell: {stats['mean_expression_per_cell']:.3f}")
    print(f"   - Mean expression per gene: {stats['mean_expression_per_gene']:.3f}")
    print(f"   - Spatial extent: X={stats['spatial_extent']['x_range']}, Y={stats['spatial_extent']['y_range']}")
    
    # Enhanced cell type prediction
    print("4. Running enhanced cell type prediction...")
    cell_type_predictor = EnhancedCellTypePredictor()
    cell_type_predictions = cell_type_predictor.predict_cell_types(data)
    cell_type_assignments = cell_type_predictor.assign_best_cell_types(cell_type_predictions)
    
    # Cell type statistics
    type_counts = {}
    confidence_stats = {}
    
    for assignment in cell_type_assignments:
        cell_type = assignment["predicted_type"]
        confidence = assignment["confidence"]
        
        type_counts[cell_type] = type_counts.get(cell_type, 0) + 1
        
        if cell_type not in confidence_stats:
            confidence_stats[cell_type] = []
        confidence_stats[cell_type].append(confidence)
    
    print("   - Cell type distribution:")
    for cell_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        mean_conf = sum(confidence_stats[cell_type]) / len(confidence_stats[cell_type])
        percentage = (count / data.n_cells) * 100
        print(f"     {cell_type}: {count} cells ({percentage:.1f}%), mean confidence: {mean_conf:.3f}")
    
    # Enhanced interaction prediction
    print("5. Predicting enhanced cell-cell interactions...")
    interaction_predictor = EnhancedInteractionPredictor()
    interactions = interaction_predictor.predict_interactions(
        data, max_distance=120.0, min_score=0.03
    )
    
    print(f"   - Found {len(interactions)} potential interactions")
    
    # Pathway enrichment analysis
    if interactions:
        print("6. Analyzing pathway enrichment...")
        pathway_enrichment = interaction_predictor.analyze_pathway_enrichment(interactions)
        
        print("   - Pathway enrichment:")
        for pathway, stats in sorted(pathway_enrichment.items(), 
                                   key=lambda x: x[1]['count'], reverse=True):
            print(f"     {pathway}: {stats['count']} interactions ({stats['frequency']*100:.1f}%), "
                  f"mean score: {stats['mean_score']:.4f}")
        
        # Show top interactions
        interactions_sorted = sorted(interactions, key=lambda x: x['interaction_score'], reverse=True)
        print("   - Top 10 interactions:")
        for i, interaction in enumerate(interactions_sorted[:10]):
            print(f"     {i+1}. Cell {interaction['sender_cell']} -> Cell {interaction['receiver_cell']}: "
                  f"{interaction['ligand']} -> {interaction['receptor']} "
                  f"(score: {interaction['interaction_score']:.4f}, "
                  f"distance: {interaction['distance']:.1f}, "
                  f"pathway: {interaction['pathway_category']})")
    
    # Spatial analysis
    print("7. Spatial neighborhood analysis...")
    neighbors = data.find_spatial_neighbors(k=8)
    
    # Calculate spatial clustering metrics
    spatial_coherence = {}
    for cell_type in type_counts.keys():
        # Find cells of this type
        type_cells = [i for i, assignment in enumerate(cell_type_assignments) 
                     if assignment["predicted_type"] == cell_type]
        
        if len(type_cells) > 1:
            # Calculate how often cells of same type are neighbors
            same_type_neighbor_count = 0
            total_neighbor_pairs = 0
            
            for cell_id in type_cells:
                for neighbor_id in neighbors.get(cell_id, []):
                    total_neighbor_pairs += 1
                    if cell_type_assignments[neighbor_id]["predicted_type"] == cell_type:
                        same_type_neighbor_count += 1
            
            coherence = same_type_neighbor_count / total_neighbor_pairs if total_neighbor_pairs > 0 else 0
            spatial_coherence[cell_type] = coherence
    
    print("   - Spatial coherence by cell type:")
    for cell_type, coherence in sorted(spatial_coherence.items(), 
                                     key=lambda x: x[1], reverse=True):
        print(f"     {cell_type}: {coherence:.3f}")
    
    # Compile comprehensive results
    print("8. Compiling comprehensive analysis results...")
    results = {
        "generation": "1_simple_enhanced",
        "analysis_metadata": {
            "n_cells_analyzed": data.n_cells,
            "n_genes_analyzed": data.n_genes,
            "analysis_type": "enhanced_comprehensive",
            "features_analyzed": [
                "cell_type_prediction",
                "interaction_prediction", 
                "pathway_enrichment",
                "spatial_coherence"
            ]
        },
        "summary_stats": stats,
        "cell_type_analysis": {
            "predictions": cell_type_predictions,
            "assignments": cell_type_assignments,
            "type_counts": type_counts,
            "confidence_stats": {
                cell_type: {
                    "mean": sum(confs) / len(confs),
                    "min": min(confs),
                    "max": max(confs)
                } for cell_type, confs in confidence_stats.items()
            },
            "spatial_coherence": spatial_coherence
        },
        "interaction_analysis": {
            "interactions": interactions[:200],  # Top 200 interactions
            "total_interactions": len(interactions),
            "pathway_enrichment": pathway_enrichment if interactions else {},
            "interaction_summary": {
                "mean_score": sum(i["interaction_score"] for i in interactions) / len(interactions) if interactions else 0,
                "mean_distance": sum(i["distance"] for i in interactions) / len(interactions) if interactions else 0,
                "unique_ligands": len(set(i["ligand"] for i in interactions)),
                "unique_receptors": len(set(i["receptor"] for i in interactions))
            }
        },
        "spatial_analysis": {
            "neighborhood_graph": {str(k): v for k, v in neighbors.items()},
            "spatial_coherence_global": sum(spatial_coherence.values()) / len(spatial_coherence) if spatial_coherence else 0
        }
    }
    
    # Save results
    with open("/root/repo/enhanced_generation1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("✅ Enhanced Generation 1 analysis complete!")
    print("   Results saved to enhanced_generation1_results.json")
    print(f"   - Analyzed {data.n_cells} cells and {data.n_genes} genes")
    print(f"   - Predicted {len(type_counts)} distinct cell types")
    print(f"   - Found {len(interactions)} cell-cell interactions")
    print(f"   - Identified {len(pathway_enrichment) if interactions else 0} active pathways")
    
    return results


if __name__ == "__main__":
    # Run the enhanced analysis
    run_enhanced_analysis()