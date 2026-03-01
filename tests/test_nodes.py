import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from utils.encoder import Encoder
from utils.decoder import Decoder

def test_encoder_active_path():
    key = jax.random.PRNGKey(0)
    encoder = Encoder(key)
    
    # Batch size 1 conceptually, but the shape given is (1, n, n) representing the single channel
    n = 16
    x = jax.random.normal(key, (1, n, n))
    is_active = jnp.array(True)
    
    out, out_active = encoder(x, is_active)
    
    assert out.shape == (8, n, n), f"Expected shape (8, {n}, {n}), got {out.shape}"
    assert out_active == True

def test_encoder_inactive_path():
    key = jax.random.PRNGKey(1)
    encoder = Encoder(key)
    
    n = 16
    x = jax.random.normal(key, (1, n, n))
    is_active = jnp.array(False)
    
    out, out_active = encoder(x, is_active)
    
    assert out.shape == (8, n, n), f"Expected shape (8, {n}, {n}), got {out.shape}"
    assert out_active == False
    assert jnp.all(out == 0.0), "Inactive encoder output must be entirely zeros."

def test_decoder_active_path():
    key = jax.random.PRNGKey(2)
    decoder = Decoder(key)
    
    n = 8
    # 16 inputs as specified
    x = jax.random.normal(key, (16, n, n))
    
    # 12 active inputs => passes gate
    is_active_flags = jnp.array([True]*12 + [False]*4)
    
    out, out_active = decoder(x, is_active_flags)
    
    assert out.shape == (1, n, n), f"Expected shape (1, {n}, {n}), got {out.shape}"
    assert out_active == True

def test_decoder_inactive_path():
    key = jax.random.PRNGKey(3)
    decoder = Decoder(key)
    
    n = 8
    x = jax.random.normal(key, (16, n, n))
    
    # Only 11 active inputs => fails gate
    is_active_flags = jnp.array([True]*11 + [False]*5)
    
    out, out_active = decoder(x, is_active_flags)
    
    assert out.shape == (1, n, n), f"Expected shape (1, {n}, {n}), got {out.shape}"
    assert out_active == False
    assert jnp.all(out == 0.0), "Inactive decoder output must be entirely zeros."

def test_decoder_jit_compilation():
    key = jax.random.PRNGKey(4)
    decoder = Decoder(key)
    
    @jax.jit
    def run_decoder(x_input, flags):
        return decoder(x_input, flags)
        
    n = 12
    x = jax.random.normal(key, (16, n, n))
    flags = jnp.array([True]*14 + [False]*2)
    
    out, out_active = run_decoder(x, flags)
    assert out.shape == (1, n, n)
    assert out_active == True
