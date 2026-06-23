#!/usr/bin/env python3
"""Merge two attacked latent caches into one."""

import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache1', type=str, required=True, help='First cache (e.g., 0-5000)')
    parser.add_argument('--cache2', type=str, required=True, help='Second cache (e.g., 5000-10000)')
    parser.add_argument('--output', type=str, required=True, help='Output merged cache')
    args = parser.parse_args()
    
    print(f"Loading: {args.cache1}")
    cache1 = torch.load(args.cache1, map_location='cpu', weights_only=False)
    
    print(f"Loading: {args.cache2}")
    cache2 = torch.load(args.cache2, map_location='cpu', weights_only=False)
    
    # Merge all tensors
    merged = {}
    for key in cache1.keys():
        if key in cache2:
            t1 = cache1[key]
            t2 = cache2[key]
            merged[key] = torch.cat([t1, t2], dim=0)
            print(f"  {key}: {t1.shape} + {t2.shape} -> {merged[key].shape}")
        else:
            merged[key] = cache1[key]
            print(f"  {key}: kept from cache1 only")
    
    # Check for keys only in cache2
    for key in cache2.keys():
        if key not in merged:
            merged[key] = cache2[key]
            print(f"  {key}: added from cache2 only")
    
    print(f"\nSaving to: {args.output}")
    torch.save(merged, args.output)
    
    size = os.path.getsize(args.output) / 1e9
    print(f"File size: {size:.2f} GB")
    print("Done!")

if __name__ == "__main__":
    main()
