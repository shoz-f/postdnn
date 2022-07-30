defmodule PostDNNTest do
  use ExUnit.Case
  doctest PostDNN

  test "non max suppression" do
    boxes  = File.read!("test/boxes.bin")
    scores = File.read!("test/scores.bin")
    with {:ok, res} = PostDNN.non_max_suppression_multi_class({43,80}, boxes, scores, boxrepr: :corner) do
      assert Map.has_key?(res, "0")
      assert Map.has_key?(res, "1")
      assert Map.has_key?(res, "2")
      assert Map.has_key?(res, "7")
      assert Map.has_key?(res, "15")
      assert Map.has_key?(res, "16")
    end
  end
  
  test "make a mesh-grid" do
    res = PostDNN.mesh_grid({416,416}, 8)
    
    assert {2704,3} = Nx.shape(res)
  end
end
