# polymer_property_predictor/models/tokenizer/polymer_tokenizer.py

import torch
from typing import List, Dict, Optional
from collections import defaultdict
import re

class ChemicalTokenizer:
    """
    A chemical-aware tokenizer for polymer SMILES strings that preserves chemical meaning
    and identifies important substructures.
    """
    def __init__(self, vocab: Optional[Dict[str, int]] = None, max_length: int = 150):
        """
        Initialize the tokenizer with vocabulary and parameters.
        
        Args:
            vocab: Optional pre-existing vocabulary mapping tokens to indices
            max_length: Maximum sequence length for padding/truncation
        """
        # Special tokens that have specific meaning in the tokenization
        self.special_tokens = {
            'PAD': '[PAD]',  # Used for padding sequences to same length
            'UNK': '[UNK]',  # Used for unknown tokens
            'CLS': '[CLS]',  # Start of sequence token
            'SEP': '[SEP]',  # End of sequence token
            'MASK': '[MASK]' # Used for masked language modeling
        }
        
        # Chemical substructure patterns for enhanced chemical understanding
        self.substructure_patterns = {
            'aromatic_atoms': ['c', 'n', 'o', 'p', 's'],
            'ring_numbers': list('12345678'),
            'branch_symbols': ['(', ')', '[', ']'],
            'bond_symbols': ['-', '=', '#', ':'],
            'polymer_markers': ['*'],
            'charges': ['+', '-'],
            'wildcards': ['X', 'R']
        }
        
        # Regular expressions for tokenization
        self.regex_patterns = {
        'atom': re.compile(r'[A-Z][a-z]?'),
        'number': re.compile(r'\d+'),
        'special': re.compile(r'[\(\)\[\]\*\+\-=#:]')
    }
        
        self.max_length = max_length
        self.vocab = self._initialize_vocabulary(vocab)
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def _initialize_vocabulary(self, vocab: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        """
        Initialize or validate the vocabulary.
        
        Args:
            vocab: Optional pre-existing vocabulary
            
        Returns:
            Complete vocabulary including special tokens and chemical tokens
        """
        if vocab is None:
            # Start with special tokens
            vocab = {token: idx for idx, token in enumerate(self.special_tokens.values())}
            
            # Add common chemical elements (first 118 elements)
            elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                       'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']  # Truncated for brevity
            
            # Add elements to vocabulary
            for element in elements:
                if element not in vocab:
                    vocab[element] = len(vocab)
            
            # Add substructure patterns
            for pattern_list in self.substructure_patterns.values():
                for pattern in pattern_list:
                    if pattern not in vocab:
                        vocab[pattern] = len(vocab)
        
        return vocab

    def _tokenize_smiles(self, smiles: str) -> List[str]:
        """
        Tokenize SMILES string into chemically meaningful tokens.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            List of chemical tokens
        """
        tokens = []
        i = 0
        
        while i < len(smiles):
            # Try matching an atom (element symbol)
            atom_match = self.regex_patterns['atom'].match(smiles[i:])
            if atom_match:
                tokens.append(atom_match.group())
                i += len(atom_match.group())
                continue
            
            # Try matching a number
            number_match = self.regex_patterns['number'].match(smiles[i:])
            if number_match:
                tokens.append(number_match.group())
                i += len(number_match.group())
                continue
            
            # Check for special characters
            special_match = self.regex_patterns['special'].match(smiles[i:])
            if special_match:
                tokens.append(special_match.group())
                i += 1
                continue
            
            # Handle other characters (like lowercase aromatic atoms)
            tokens.append(smiles[i])
            i += 1
            
        return tokens

    def _identify_substructures(self, tokens: List[str]) -> Dict[str, List[int]]:
        """
        Identify chemical substructures in the token sequence.
        
        Args:
            tokens: List of chemical tokens
            
        Returns:
            Dictionary mapping substructure types to token positions
        """
        substructures = defaultdict(list)
        
        for i, token in enumerate(tokens):
            # Identify aromatic atoms
            if token in self.substructure_patterns['aromatic_atoms']:
                substructures['aromatic'].append(i)
            
            # Identify ring markers
            elif token in self.substructure_patterns['ring_numbers']:
                substructures['ring'].append(i)
            
            # Identify branch points
            elif token in self.substructure_patterns['branch_symbols']:
                substructures['branch'].append(i)
            
            # Identify special bonds
            elif token in self.substructure_patterns['bond_symbols']:
                substructures['bond'].append(i)
            
            # Identify polymer markers
            elif token in self.substructure_patterns['polymer_markers']:
                substructures['polymer'].append(i)
        
        return substructures

    def encode(self, 
              smiles: str, 
              add_special_tokens: bool = True,
              padding: bool = True,
              return_substructures: bool = False) -> Dict[str, torch.Tensor]:
        """
        Encode SMILES string into token indices and attention masks.
        
        Args:
            smiles: Input SMILES string
            add_special_tokens: Whether to add CLS and SEP tokens
            padding: Whether to pad sequence to max_length
            return_substructures: Whether to return substructure attention masks
            
        Returns:
            Dictionary containing token indices and attention masks
        """
        # Tokenize SMILES
        tokens = self._tokenize_smiles(smiles)
        
        # Convert tokens to indices
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.vocab[self.special_tokens['CLS']])
        
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.vocab[self.special_tokens['UNK']]))
        
        if add_special_tokens:
            token_ids.append(self.vocab[self.special_tokens['SEP']])
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)
        
        # Pad if necessary
        if padding:
            pad_length = self.max_length - len(token_ids)
            if pad_length > 0:
                token_ids.extend([self.vocab[self.special_tokens['PAD']]] * pad_length)
                attention_mask.extend([0] * pad_length)
            else:
                token_ids = token_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
        
        # Convert to tensors
        encoded = {
            'input_ids': torch.tensor(token_ids),
            'attention_mask': torch.tensor(attention_mask)
        }
        
        # Add substructure information if requested
        if return_substructures:
            substructures = self._identify_substructures(tokens)
            for subtype, positions in substructures.items():
                mask = torch.zeros(len(token_ids))
                mask[positions] = 1
                encoded[f'{subtype}_mask'] = mask
        
        return encoded

    def batch_encode(self, 
                    smiles_list: List[str],
                    add_special_tokens: bool = True,
                    padding: bool = True,
                    return_substructures: bool = False) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            add_special_tokens: Whether to add CLS and SEP tokens
            padding: Whether to pad sequences
            return_substructures: Whether to return substructure attention masks
            
        Returns:
            Dictionary containing batched token indices and attention masks
        """
        encoded_list = [self.encode(smiles, add_special_tokens, padding, return_substructures) 
                       for smiles in smiles_list]
        
        # Batch all tensors
        batched = {
            key: torch.stack([e[key] for e in encoded_list])
            for key in encoded_list[0].keys()
        }
        
        return batched

    def decode(self, 
               token_ids: torch.Tensor,
               skip_special_tokens: bool = True) -> str:
        """
        Decode token indices back to SMILES string.
        
        Args:
            token_ids: Tensor of token indices
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded SMILES string
        """
        tokens = []
        for idx in token_ids:
            token = self.inverse_vocab[idx.item()]
            if skip_special_tokens and token in self.special_tokens.values():
                continue
            tokens.append(token)
        
        # Post-process to ensure valid SMILES
        smiles = ''.join(tokens)
        return smiles

    def save_vocabulary(self, path: str):
        """Save the tokenizer vocabulary to a file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.vocab, f, indent=2)
    
    @classmethod
    def from_vocabulary(cls, path: str, max_length: int = 150):
        """Load a tokenizer from a saved vocabulary file."""
        import json
        with open(path, 'r') as f:
            vocab = json.load(f)
        return cls(vocab=vocab, max_length=max_length)
