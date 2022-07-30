defmodule PostDNN.NIF do
  @moduledoc false
  @compile {:autoload, false}

  #loading NIF library
  @on_load :load_nif
  def load_nif do
    nif_file = Application.app_dir(:postdnn, "priv/postdnn_nif")
    :erlang.load_nif(nif_file, 0)
  end

  # stub implementations for NIFs (fallback)
  def dnn_non_max_suppression_multi_class(_1, _2, _3, _4, _5, _6, _7, _8),
    do: raise("NIF dnn_non_max_suppression_multi_class/8 not implemented")
end
