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
    
    assert out.shape == (n, n), f"Expected shape ({n}, {n}), got {out.shape}"
    assert out_active == True

def test_decoder_inactive_path():
    key = jax.random.PRNGKey(3)
    decoder = Decoder(key)
    
    n = 8
    x = jax.random.normal(key, (16, n, n))
    
    # Only 11 active inputs => fails gate
    is_active_flags = jnp.array([True]*11 + [False]*5)
    
    out, out_active = decoder(x, is_active_flags)
    
    assert out.shape == (n, n), f"Expected shape ({n}, {n}), got {out.shape}"
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
    assert out.shape == (n, n)
    assert out_active == True


# ─── Encoder node — tree branch behaviour ────────────────────────────────────

def test_encoder_nonzero_active_output():
    """Active path with a non-trivial input must produce a non-zero output."""
    key = jax.random.PRNGKey(5)
    encoder = Encoder(key)
    n = 8
    x = jax.random.normal(key, (1, n, n))
    out, out_active = encoder(x, jnp.array(True))
    assert bool(jnp.any(out != 0.0))

def test_encoder_output_shape_custom_channels():
    """Encoder with in_c=8, out_c=8 (leaf config) returns (8, n, n)."""
    from utils.config.encode import ENCODER_OUT_CHANNELS
    key = jax.random.PRNGKey(6)
    encoder = Encoder(key, in_channels=1, out_channels=ENCODER_OUT_CHANNELS)
    n = 10
    x = jax.random.normal(key, (1, n, n))
    out, active = encoder(x, jnp.array(True))
    assert out.shape == (ENCODER_OUT_CHANNELS, n, n)
    assert bool(active)

def test_encoder_deterministic():
    """Same key + input must give bit-identical output."""
    key = jax.random.PRNGKey(7)
    encoder = Encoder(key)
    n = 6
    x = jax.random.normal(key, (1, n, n))
    out1, _ = encoder(x, jnp.array(True))
    out2, _ = encoder(x, jnp.array(True))
    assert bool(jnp.allclose(out1, out2))

def test_encoder_different_inputs_differ():
    """Two distinct inputs produce distinct outputs from the same encoder."""
    key = jax.random.PRNGKey(8)
    encoder = Encoder(key)
    n = 6
    k1, k2 = jax.random.split(key)
    x1 = jax.random.normal(k1, (1, n, n))
    x2 = jax.random.normal(k2, (1, n, n))
    out1, _ = encoder(x1, jnp.array(True))
    out2, _ = encoder(x2, jnp.array(True))
    assert not bool(jnp.allclose(out1, out2))

def test_encoder_zero_input_active():
    """Zero input through an active encoder: output shape correct, flag True."""
    key = jax.random.PRNGKey(9)
    encoder = Encoder(key)
    n = 8
    x = jnp.zeros((1, n, n))
    out, active = encoder(x, jnp.array(True))
    assert out.shape == (8, n, n)
    assert bool(active)
